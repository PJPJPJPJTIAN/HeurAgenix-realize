import numpy as np  # 导入numpy库，用于数值计算（如距离矩阵处理）
import networkx as nx  # 导入networkx库，用于处理图结构（TSP问题可抽象为图问题）
import tsplib95  # 导入tsplib95库，用于加载TSPLIB格式的问题实例数据
from src.problems.base.env import BaseEnv  # 从基础模块导入BaseEnv类，作为TSP环境的父类
from src.problems.tsp.components import Solution  # 从TSP组件模块导入Solution类，用于存储TSP解

class Env(BaseEnv):  # 定义TSP环境类，继承自BaseEnv，用于管理TSP问题的实例数据、当前解和问题状态
    """TSP env that stores the instance data, current solution, and problem state to support algorithm."""  # 类文档字符串：说明该环境存储实例数据、当前解和问题状态，用于支持启发式算法
    def __init__(self, data_name: str, **kwargs):  # 初始化方法，接收数据名称和其他关键字参数
        super().__init__(data_name, "tsp")  # 调用父类BaseEnv的初始化方法，传入数据名称和问题类型"tsp"
        self.construction_steps = self.instance_data["node_num"]  # 定义构建完整解所需的步骤数为节点数量（每个步骤添加一个节点）
        self.key_item = "current_cost"  # 定义评估解质量的关键指标为"current_cost"（当前路径成本）
        self.compare = lambda x, y: y - x  # 定义比较函数：若y > x则返回正值，用于判断解的优劣（成本越小越好）

    @property
    def is_complete_solution(self) -> bool:  # 定义属性方法，判断当前解是否为完整解
        return len(set(self.current_solution.tour)) == self.instance_data["node_num"]  # 若路径中节点去重后的数量等于总节点数，则为完整解（每个节点都被访问且不重复）

    def load_data(self, data_path: str) -> None:  # 定义加载数据的方法，接收数据路径并返回处理后的实例数据
        problem = tsplib95.load(data_path)  # 使用tsplib95加载TSPLIB格式的问题文件
        distance_matrix = nx.to_numpy_array(problem.get_graph())  # 将问题转换为图结构，再转为距离矩阵（numpy数组）
        node_num = len(distance_matrix)  # 计算节点数量（距离矩阵的维度）
        return {"node_num": node_num, "distance_matrix": distance_matrix}  # 返回包含节点数和距离矩阵的字典

    def init_solution(self) -> None:  # 定义初始化解的方法
        return Solution(tour=[])  # 返回一个空路径的Solution实例（初始解为空）

    def get_key_value(self, solution: Solution=None) -> float:  # 定义获取关键指标（当前成本）的方法，可选参数为指定解（默认使用当前解）
        """Get the key value of the current solution based on the key item."""  # 方法文档字符串：说明该方法基于关键指标获取当前解的值
        if solution is None:  # 若未指定解，则使用环境中的当前解
            solution = self.current_solution
        current_cost = sum([self.instance_data["distance_matrix"][solution.tour[index]][solution.tour[index + 1]] for index in range(len(solution.tour) - 1)])  # 计算路径中相邻节点的距离总和
        if len(solution.tour) > 0:  # 若路径非空，需加上从最后一个节点返回第一个节点的距离（形成回路）
            current_cost += self.instance_data["distance_matrix"][solution.tour[-1]][solution.tour[0]]
        return current_cost  # 返回计算得到的总路径成本

    def validation_solution(self, solution: Solution=None) -> bool:  # 定义验证解有效性的方法，可选参数为指定解（默认使用当前解）
        """
        Check the validation of this solution in following items:  # 方法文档字符串：说明验证解的三个维度
            1. Node Existence: Each node in the solution must exist within the problem instance's range of nodes.  # 节点存在性：路径中的节点必须在实例的节点范围内
            2. Uniqueness: No node is repeated within the solution path, ensuring that each node is visited at most once.  # 唯一性：路径中节点不重复（每个节点最多访问一次）
            3. Connectivity: Each edge (from one node to the next) must be connected, i.e., not marked as infinite distance in the distance matrix.  # 连通性：相邻节点间的边必须存在（距离不为无穷大）
        """
        node_set = set()  # 用于存储已出现的节点，检查唯一性
        if solution is None:  # 若未指定解，则使用环境中的当前解
            solution = self.current_solution

        if not isinstance(solution, Solution) or not isinstance(solution.tour, list):  # 检查解是否为Solution实例且路径为列表类型
            return False
        if solution is not None and solution.tour is not None:  # 若解和路径存在，则进行详细验证
            for index, node in enumerate(solution.tour):  # 遍历路径中的每个节点
                # Check node existence 检查节点存在性：节点索引必须在[0, 节点数)范围内
                if not (0 <= node < self.instance_data["node_num"]):
                    return False

                # Check uniqueness 检查唯一性：节点不能已在node_set中（避免重复）
                if node in node_set:
                    return False
                node_set.add(node)  # 将当前节点加入node_set

                # Check connectivity if not the last node 检查连通性（非最后一个节点）：当前节点与下一个节点的距离不能为无穷大
                if index < len(solution.tour) - 1:
                    next_node = solution.tour[index + 1]
                    if self.instance_data["distance_matrix"][node][next_node] == np.inf:
                        return False
        return True  # 所有检查通过，返回True（解有效）