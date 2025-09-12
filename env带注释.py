import os  # 导入os模块，用于处理文件路径和目录操作
import traceback  # 导入traceback模块，用于捕获和打印异常堆栈信息
from src.problems.base.components import BaseSolution, BaseOperator  # 从基础组件模块导入BaseSolution（解决方案基类）和BaseOperator（操作基类）
from src.util.util import load_function, search_file  # 从工具模块导入load_function（加载函数）和search_file（搜索文件）工具函数


class BaseEnv:  # 定义基础环境类BaseEnv，作为问题求解环境的基类
    """Base env that stores the static global data, current solution, dynamic state and provide necessary to support algorithm."""  # 文档字符串：基础环境存储静态全局数据、当前解决方案、动态状态，并为算法提供必要支持
    def __init__(self, data_name: str, problem: str, **kwargs):  # 初始化方法，接收数据名称、问题名称和其他关键字参数
        self.problem = problem  # 存储问题名称
        self.data_path = search_file(data_name, problem)  # 调用search_file工具函数查找数据文件路径，并存储
        self.data_ref_name = data_name.split(os.sep)[-1]  # 从数据名称中提取最后一部分作为数据引用名（通常是文件名）
        assert self.data_path is not None  # 断言数据路径不为空，确保找到数据文件
        self.instance_data: tuple = self.load_data(self.data_path)  # 调用load_data方法加载数据文件内容，存储为实例数据（元组类型）
        self.current_solution: BaseSolution = self.init_solution()  # 调用init_solution方法初始化当前解决方案（BaseSolution类型）
        self.algorithm_data: dict = None  # 初始化算法数据为None（用于存储算法运行过程中的动态数据）
        self.recordings: list[tuple] = None  # 初始化记录列表为None（用于记录算法运行轨迹）
        self.output_dir: str = None  # 初始化输出目录为None（用于指定结果保存路径）
        # Maximum step to constructive a complete solution
        self.construction_steps: int = None  # 构建完整解决方案的最大步骤数，初始化为None
        # Key item in state to compare the solution
        self.key_item: str = None  # 用于比较解决方案的关键指标名称，初始化为None
        # Returns the advantage of the first and second key value 
        # A return value greater than 0 indicates that first is better and the larger the number, the greater the advantage.
        self.compare: callable = None  # 用于比较两个关键值优劣的可调用对象，返回值>0表示前者更优，初始化为None

        problem_state_file = search_file("problem_state.py", problem=self.problem)  # 查找当前问题的problem_state.py文件路径
        assert problem_state_file is not None, f"Problem state code file {problem_state_file} does not exist"  # 断言找到该文件，否则抛出错误
        self.get_instance_problem_state = load_function(problem_state_file, problem=self.problem, function_name="get_instance_problem_state")  # 加载获取实例问题状态的函数
        self.get_solution_problem_state = load_function(problem_state_file, problem=self.problem, function_name="get_solution_problem_state")  # 加载获取解决方案问题状态的函数
        self.problem_state = self.get_problem_state()  # 调用get_problem_state方法获取当前问题状态并存储


    @property
    def is_complete_solution(self) -> bool:  # 定义is_complete_solution属性，用于判断当前解决方案是否完整
        pass  # 基类中未实现，需子类重写

    @property
    def is_valid_solution(self) -> bool:  # 定义is_valid_solution属性，用于判断当前解决方案是否有效
        return self.validation_solution(self.current_solution)  # 调用validation_solution方法验证当前解决方案

    @property
    def continue_run(self) -> bool:  # 定义continue_run属性，用于判断是否继续运行算法
        return True  # 基类默认返回True，需子类根据实际情况重写

    @property
    def key_value(self) -> float:  # 定义key_value属性，用于获取当前解决方案的关键指标值
        """Get the key value of the current solution."""  # 文档字符串：获取当前解决方案的关键值
        return self.get_key_value(self.current_solution)  # 调用get_key_value方法获取当前解决方案的关键值

    def get_key_value(self, solution: BaseSolution=None) -> float:  # 定义获取解决方案关键值的方法，接收解决方案参数（默认当前解决方案）
        """Get the key value of the solution."""  # 文档字符串：获取解决方案的关键值
        pass  # 基类中未实现，需子类重写

    def reset(self, output_dir: str=None):  # 定义重置环境的方法，接收输出目录参数（可选）
        self.current_solution = self.init_solution()  # 重新初始化当前解决方案
        self.problem_state = self.get_problem_state()  # 重新获取问题状态
        self.algorithm_data = {}  # 重置算法数据为空字典
        self.recordings = []  # 重置记录列表为空列表
        if output_dir:  # 如果指定了输出目录
            if os.sep in output_dir:  # 如果输出目录包含路径分隔符（即绝对路径或相对路径）
                self.output_dir = output_dir  # 直接使用该路径作为输出目录
            else:  # 否则，构建输出目录路径
                base_output_dir = os.path.join(os.getenv("AMLT_OUTPUT_DIR"), "..", "..", "output") if os.getenv("AMLT_OUTPUT_DIR") else "output"  # 基础输出目录：优先使用环境变量AMLT_OUTPUT_DIR，否则使用"output"
                self.output_dir = os.path.join(base_output_dir, self.problem, "result", self.data_ref_name, output_dir)  # 拼接完整输出目录路径：基础目录/问题名/result/数据引用名/指定目录
            os.makedirs(self.output_dir, exist_ok=True)  # 创建输出目录（如果不存在）

    def load_data(self, data_path: str) -> dict:  # 定义加载数据的方法，接收数据路径参数，返回字典
        pass  # 基类中未实现，需子类重写

    def init_solution(self) -> None:  # 定义初始化解决方案的方法，返回None
        pass  # 基类中未实现，需子类重写

    def helper_function(self) -> dict:  # 定义辅助函数方法，返回包含辅助函数的字典
        return {"get_problem_state": self.get_problem_state, "validation_solution": self.validation_solution}  # 返回包含get_problem_state和validation_solution方法的字典

    def get_problem_state(self, solution: BaseSolution=None) -> dict:  # 定义获取问题状态的方法，接收解决方案参数（默认当前解决方案）
        if solution is None:  # 如果未指定解决方案
            solution = self.current_solution  # 使用当前解决方案
        instance_problem_state = self.get_instance_problem_state(self.instance_data)  # 调用get_instance_problem_state获取实例问题状态
        solution_problem_state = self.get_solution_problem_state(self.instance_data, solution)  # 调用get_solution_problem_state获取解决方案问题状态
        helper_function = self.helper_function()  # 获取辅助函数字典
        problem_state = None  # 初始化问题状态为None
        if solution_problem_state:  # 如果解决方案问题状态存在
            problem_state = {  # 构建问题状态字典
                **self.instance_data,  # 包含实例数据
                "current_solution": solution,  # 包含当前解决方案
                self.key_item: self.key_value,  # 包含关键指标值
                **helper_function,  # 包含辅助函数
                **instance_problem_state,  # 包含实例问题状态
                **solution_problem_state,  # 包含解决方案问题状态
            }
        return problem_state  # 返回构建的问题状态

    def validation_solution(self, solution: BaseSolution=None) -> bool:  # 定义验证解决方案的方法，接收解决方案参数（默认当前解决方案）
        """Check the validation of this solution"""  # 文档字符串：检查解决方案的有效性
        pass  # 基类中未实现，需子类重写

    def run_heuristic(self, heuristic: callable, parameters:dict={}, add_record_item: dict={}) -> BaseOperator:  # 定义运行启发式算法的方法，接收启发式函数、参数字典、附加记录项，返回BaseOperator实例
        try:  # 捕获异常
            operator, delta = heuristic(  # 调用启发式函数，传入问题状态、算法数据和参数，返回操作符和算法数据更新量
                problem_state=self.problem_state,
                algorithm_data=self.algorithm_data,** parameters
            )
            if isinstance(operator, BaseOperator):  # 如果返回的操作符是BaseOperator的实例
                self.run_operator(operator)  # 调用run_operator方法执行该操作
                self.algorithm_data.update(delta)  # 更新算法数据
            record_item = {"operation_id": len(self.recordings), "heuristic": heuristic.__name__, "operator": operator}  # 构建记录项：操作ID、启发式函数名、操作符
            record_item.update(add_record_item)  # 添加附加记录项
            self.recordings.append(record_item)  # 将记录项添加到记录列表
            return operator  # 返回操作符
        except Exception as e:  # 捕获异常
            trace_string = traceback.format_exc()  # 获取异常堆栈信息
            print(trace_string)  # 打印异常信息
            return trace_string  # 返回异常信息字符串

    def run_operator(self, operator: BaseOperator) -> bool:  # 定义执行操作的方法，接收BaseOperator实例，返回布尔值
        if isinstance(operator, BaseOperator):  # 如果操作符是BaseOperator的实例
            self.current_solution = operator.run(self.current_solution)  # 调用操作符的run方法更新当前解决方案
            self.problem_state = self.get_problem_state()  # 重新获取问题状态
        return operator  # 返回操作符（原文返回值类型标注为bool，但实际返回operator，可能是笔误）

    def summarize_env(self) -> str:  # 定义环境总结方法，返回字符串
        pass  # 基类中未实现，需子类重写

    def __getstate__(self):  # 定义序列化方法，用于pickle时保存对象状态
        state = self.__dict__.copy()  # 复制当前对象的属性字典
        state.pop("get_instance_problem_state", None)  # 移除get_instance_problem_state方法（不可序列化）
        state.pop("get_solution_problem_state", None)  # 移除get_solution_problem_state方法（不可序列化）
        return state  # 返回处理后的状态字典

    def __setstate__(self, state):  # 定义反序列化方法，用于unpickle时恢复对象状态
        self.__dict__.update(state)  # 恢复属性字典
        self.get_instance_problem_state = load_function("problem_state.py", problem=self.problem, function_name="get_instance_problem_state")  # 重新加载get_instance_problem_state方法
        self.get_solution_problem_state = load_function("problem_state.py", problem=self.problem, function_name="get_solution_problem_state")  # 重新加载get_solution_problem_state方法

    def dump_result(self, content_dict: dict={}, dump_records: list=["operation_id", "operator", "heuristic"], result_file: str="result.txt") -> str:  # 定义保存结果的方法，接收内容字典、需 dump 的记录项列表、结果文件名，返回内容字符串
        content = f"-data: {self.data_path}\n"  # 构建结果内容：数据路径
        content += f"-current_solution:\n{self.current_solution}\n"  # 添加上当前解决方案
        content += f"-is_complete_solution: {self.is_complete_solution}\n"  # 添加解决方案是否完整的标志
        content += f"-is_valid_solution: {self.is_valid_solution}\n"  # 添加解决方案是否有效的标志
        content += f"-{self.key_item}: {self.key_value}\n"  # 添加关键指标名称和值
        for item, value in content_dict.items():  # 遍历内容字典，添加额外内容
            content += f"-{item}: {value}\n"
        if dump_records and len(dump_records) > 0 and len(self.recordings) > 0 and len(self.recordings[0].keys()) > 0:  # 如果需要dump记录且记录存在
            dump_records = [item for item in dump_records if item in self.recordings[0].keys()]  # 筛选出记录中实际存在的项
            content += "-trajectory:\n" + "\t".join(dump_records) + "\n"  # 添加轨迹标题行（记录项名称）
            trajectory_str = "\n".join([  # 构建轨迹内容字符串：每行对应一条记录，记录项用制表符分隔
                "\t".join([str(recording_item.get(item, "None")) for item in dump_records])
                for recording_item in self.recordings
            ])
            content += trajectory_str  # 添加轨迹内容

        if self.output_dir != None and result_file != None:  # 如果指定了输出目录和结果文件
            output_file = os.path.join(self.output_dir, result_file)  # 构建输出文件路径
            with open(output_file, "w") as file:  # 打开文件
                file.write(content)  # 写入内容
        
        return content  # 返回结果内容字符串