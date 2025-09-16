import os  # 导入os模块，用于处理文件和目录相关操作
import random  # 导入random模块，用于生成随机数，实现随机选择功能
from src.problems.base.components import BaseOperator  # 从基础组件模块导入BaseOperator类，可能用于类型提示或基础操作定义
from src.problems.base.env import BaseEnv  # 从基础环境模块导入BaseEnv类，作为环境的基类，提供问题运行的环境接口
from src.util.util import load_function  # 从工具模块导入load_function函数，用于加载启发式函数


class RandomHyperHeuristic:  # 定义随机超启发式类，用于通过随机选择启发式算法解决问题
    def __init__(  # 类的初始化方法，用于设置超启发式的参数并加载启发式函数池
        self,
        heuristic_pool: list[str],  # 启发式算法池，包含可用启发式的文件名列表
        problem: str,  # 问题名称，用于定位相关文件和配置
        iterations_scale_factor: float=2.0,  # 迭代缩放因子，用于计算最大步骤数
    ) -> None:
        self.heuristic_pools = [load_function(heuristic, problem=problem) for heuristic in heuristic_pool]  
        # 加载启发式池中的所有函数，存储为函数对象列表
        self.iterations_scale_factor = iterations_scale_factor  # 保存迭代缩放因子

    def run(self, env:BaseEnv) -> bool:  # 定义运行方法，接收环境实例并返回是否成功得到有效完整解
        max_steps = int(env.construction_steps * self.iterations_scale_factor) 
        # 计算最大步骤数，为环境构建步骤数乘以缩放因子
        current_steps = 0  # 初始化当前步骤数为0
        while current_steps <= max_steps and env.continue_run:  
            # 循环执行，直到达到最大步骤或环境停止运行
            heuristic = random.choice(self.heuristic_pools)  # 从启发式池中随机选择一个启发式函数
            _ = env.run_heuristic(heuristic) 
            # 在环境中运行选中的启发式函数，忽略返回结果
            current_steps += 1  # 当前步骤数加1
        return env.is_complete_solution and env.is_valid_solution  # 返回是否得到完整且有效的解决方案