from src.problems.base.components import BaseSolution, BaseOperator  # 从基础组件模块导入基类BaseSolution和BaseOperator，用于继承实现TSP问题的解决方案和操作符
class Solution(BaseSolution):  # 定义TSP问题的解决方案类，继承自BaseSolution
    """The solution of TSP.  # 文档字符串：TSP问题的解决方案
    A list of integers where each integer represents a node (city) in the TSP tour.  # 解决方案是一个整数列表，每个整数代表TSP路径中的一个节点（城市）
    The order of the nodes in the list defines the order in which the cities are visited in the tour.  # 列表中节点的顺序定义了路径中城市的访问顺序
    """
    def __init__(self, tour: list[int]):  # 初始化方法，接收一个表示路径的整数列表tour
        self.tour = tour  # 将输入的tour存储为实例变量，代表TSP的访问路径

    def __str__(self) -> str:  # 重写字符串方法，用于打印解决方案
        if len(self.tour) > 0:  # 如果路径不为空
            return "tour: " + "->".join(map(str, self.tour + [self.tour[0]]))  # 返回路径字符串，格式为"tour: 节点1->节点2->...->节点n->节点1"（体现回到起点）
        return "tour: "  # 如果路径为空，返回"tour: "


class AppendOperator(BaseOperator):  # 定义追加操作符类，继承自BaseOperator，用于在路径末尾添加节点
    """Append the node at the end of the solution."""  # 文档字符串：在解决方案的末尾追加节点
    def __init__(self, node: int):  # 初始化方法，接收要追加的节点node
        self.node = node  # 存储节点为实例变量

    def run(self, solution: Solution) -> Solution:  # 执行操作的方法，接收当前解决方案，返回新的解决方案
        new_tour = solution.tour + [self.node]  # 生成新路径：在当前路径末尾添加节点
        return Solution(new_tour)  # 返回包含新路径的Solution实例


class InsertOperator(BaseOperator):  # 定义插入操作符类，继承自BaseOperator，用于在指定位置插入节点
    """Insert the node into the solution at the target position."""  # 文档字符串：在目标位置插入节点到解决方案中
    def __init__(self, node: int, position: int):  # 初始化方法，接收要插入的节点node和插入位置position
        self.node = node  # 存储节点
        self.position = position  # 存储位置

    def run(self, solution: Solution) -> Solution:  # 执行操作的方法
        new_tour = solution.tour[:self.position] + [self.node] + solution.tour[self.position:]  # 生成新路径：在指定位置插入节点
        return Solution(new_tour)  # 返回新的Solution实例


class SwapOperator(BaseOperator):  # 定义交换操作符类，继承自BaseOperator，用于交换路径中的节点对
    """Swap two nodes in the solution. swap_node_pairs is a list of tuples, each containing the two nodes to swap."""  # 文档字符串：交换解决方案中的两个节点，swap_node_pairs是元组列表，每个元组包含要交换的两个节点
    def __init__(self, swap_node_pairs: list[tuple[int, int]]):  # 初始化方法，接收要交换的节点对列表
        self.swap_node_pairs = swap_node_pairs  # 存储节点对列表

    def run(self, solution: Solution) -> Solution:  # 执行操作的方法
        node_to_index = {node: index for index, node in enumerate(solution.tour)}  # 创建节点到索引的映射字典，方便查找节点位置
        new_tour = solution.tour.copy()  # 复制当前路径作为新路径的基础
        for node_a, node_b in self.swap_node_pairs:  # 遍历每个节点对
            index_a = node_to_index.get(node_a)  # 获取节点a的索引
            index_b = node_to_index.get(node_b)  # 获取节点b的索引
            assert index_a is not None  # 断言节点a存在于路径中
            assert index_b is not None  # 断言节点b存在于路径中
            new_tour[index_a], new_tour[index_b] = new_tour[index_b], new_tour[index_a]  # 交换两个节点的位置
        return Solution(new_tour)  # 返回新的Solution实例


class ReplaceOperator(BaseOperator):  # 定义替换操作符类，继承自BaseOperator，用于替换路径中的节点
    """Replace a node with another one in the solution."""  # 文档字符串：用另一个节点替换解决方案中的某个节点
    def __init__(self, node: int, new_node: int):  # 初始化方法，接收要被替换的节点node和新节点new_node
        self.node = node  # 存储旧节点
        self.new_node = new_node  # 存储新节点

    def run(self, solution: Solution) -> Solution:  # 执行操作的方法
        index = solution.tour.index(self.node)  # 查找旧节点在路径中的索引
        new_tour = solution.tour[:index] + [self.new_node] + solution.tour[index + 1:]  # 生成新路径：用新节点替换旧节点
        return Solution(new_tour)  # 返回新的Solution实例


class ReverseSegmentOperator(BaseOperator):  # 定义反转片段操作符类，继承自BaseOperator，用于反转路径中的多个片段
    """Reverse multiple segments of indices in the solution."""  # 文档字符串：反转解决方案中索引的多个片段
    def __init__(self, segments: list[tuple[int, int]]):  # 初始化方法，接收要反转的片段列表（每个片段是起始和结束索引的元组）
        self.segments = segments  # 存储片段列表

    def run(self, solution: Solution) -> Solution:  # 执行操作的方法
        new_tour = solution.tour[:]  # 复制当前路径作为新路径的基础

        for segment in self.segments:  # 遍历每个片段
            start_index, end_index = segment  # 解包片段的起始和结束索引

            # Ensure the indices are within the bounds of the tour list  # 确保索引在路径列表的范围内
            assert 0 <= start_index < len(new_tour)  # 断言起始索引有效
            assert 0 <= end_index < len(new_tour)  # 断言结束索引有效

            if start_index <= end_index:  # 如果起始索引小于等于结束索引（正常顺序的片段）
                # Reverse the segment between start_index and end_index (inclusive)  # 反转从start_index到end_index（包含）的片段
                new_tour[start_index:end_index + 1] = reversed(new_tour[start_index:end_index + 1])
            else:  # 如果起始索引大于结束索引（跨路径首尾的片段）
                # Reverse the segment outside start_index and end_index (inclusive)  # 反转start_index到末尾和开头到end_index（包含）的组合片段
                new_tour = list(reversed(new_tour[start_index:])) + new_tour[end_index + 1:start_index] + list(reversed(new_tour[:end_index + 1]))

        return Solution(new_tour)  # 返回新的Solution实例