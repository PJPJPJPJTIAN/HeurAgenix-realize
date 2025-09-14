import os  # 导入os模块，用于处理文件路径等操作系统相关功能
import random  # 导入random模块，用于生成随机数，实现扰动概率控制
from src.problems.base.components import BaseOperator  # 从基础组件模块导入BaseOperator类，作为所有操作的基类
from src.problems.base.env import BaseEnv  # 从基础环境模块导入BaseEnv类，作为问题环境的基类
from src.util.util import load_function  # 从工具模块导入load_function函数，用于加载启发式函数


class PerturbationHyperHeuristic:  # 定义PerturbationHyperHeuristic类，用于实现带扰动的超启发式算法（结合主启发式和扰动启发式）
    def __init__(  # 类的初始化方法，用于设置超启发式的核心参数
        self,
        main_heuristic_file: str,  # 主启发式算法的文件路径
        perturbation_heuristic_file: str,  # 扰动启发式算法的文件路径
        problem: str,  # 问题名称，用于定位相关资源
        perturbation_ratio: float=0.1,  # 扰动概率（默认0.1），即选择扰动启发式的概率
        iterations_scale_factor: float = 2.0  # 迭代次数缩放因子（默认2.0），用于计算最大运行步数
    ) -> None:
        self.main_heuristic = load_function(main_heuristic_file, problem=problem)  # 加载主启发式函数并存储在实例变量中
        self.perturbation_heuristic = load_function(perturbation_heuristic_file, problem=problem)  # 加载扰动启发式函数并存储在实例变量中
        self.perturbation_ratio = perturbation_ratio  # 存储扰动概率参数
        self.iterations_scale_factor = iterations_scale_factor  # 存储迭代次数缩放因子参数

    def run(self, env:BaseEnv) -> bool:  # 定义run方法，用于在环境中运行带扰动的超启发式算法，返回布尔值表示是否得到有效完整解
        max_steps = int(env.construction_steps * self.iterations_scale_factor)  # 计算最大运行步数：环境默认构建步数 × 缩放因子（转为整数）
        current_steps = 0  # 初始化当前步骤计数为0
        heuristic_work = BaseOperator()  # 初始化启发式操作结果为BaseOperator实例，用于启动循环
        # 循环条件：当前步骤未超过最大步数、操作结果是BaseOperator实例、环境允许继续运行
        while current_steps <= max_steps and isinstance(heuristic_work, BaseOperator) and env.continue_run:
            if random.random() < self.perturbation_ratio:  # 生成[0,1)随机数，若小于扰动概率，则选择扰动启发式
                heuristic = self.perturbation_heuristic
            else:  # 否则选择主启发式
                heuristic = self.main_heuristic
            heuristic_work = env.run_heuristic(heuristic)  # 调用环境的run_heuristic方法执行选中的启发式，获取操作结果
            current_steps += 1  # 步骤计数加1
        return env.is_complete_solution and env.is_valid_solution  # 返回环境中解的完整性和有效性的布尔值判断结果（两者都为真则返回真）