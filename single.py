import os  # 导入os模块，用于处理文件路径等操作系统相关功能
from src.problems.base.componentsimport BaseOperator  # 从基础组件模块导入BaseOperator类，作为所有操作的基类
from src.problems.base.env import BaseEnv  # 从基础环境模块导入BaseEnv类，作为问题环境的基类
from src.util.util import load_function  # 从工具模块导入load_function函数，用于加载启发式函数


class SingleHyperHeuristic:  # 定义SingleHyperHeuristic类，用于单一启发式算法的执行控制
    def __init__(  # 类的初始化方法
        self,
        heuristic: str,  # 启发式算法的文件路径或代码字符串
        problem: str,  # 问题名称，用于定位相关资源
    ) -> None:
        self.heuristic = load_function(heuristic, problem=problem)  # 加载启发式函数并存储在实例变量中

    def run(self, env: BaseEnv, **kwargs) -> bool:  # 定义run方法，用于在环境中运行启发式算法，返回布尔值表示是否得到有效完整解
        current_steps = 0  # 初始化当前步骤计数为0
        heuristic_work = BaseOperator()  # 初始化启发式操作结果为BaseOperator实例，用于启动循环
        while isinstance(heuristic_work, BaseOperator) and env.continue_run:  # 当操作结果是BaseOperator实例且环境允许继续运行时，循环执行
            heuristic_work = env.run_heuristic(self.heuristic)  # 调用环境的run_heuristic方法执行启发式算法，获取操作结果
            current_steps += 1  # 步骤计数加1
        return env.is_complete_solution and env.is_valid_solution  # 返回环境中解的完整性和有效性的布尔值判断结果（两者都为真则返回真）