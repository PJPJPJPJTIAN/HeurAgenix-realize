import os  # 导入os模块，用于处理文件和目录操作
import importlib  # 导入importlib模块，用于动态导入模块
import re  # 导入re模块，用于正则表达式操作
import pandas as pd  # 导入pandas库，用于数据处理和分析
import traceback  # 导入traceback模块，用于捕获和打印异常堆栈信息
from io import StringIO  # 从io模块导入StringIO，用于字符串与文件对象的转换
from src.problems.base.env import BaseEnv 
# 从指定路径导入BaseEnv类，作为环境基类
from src.pipeline.heuristic_generator import HeuristicGenerator  
# 导入启发式生成器类，用于生成启发式算法
from src.pipeline.hyper_heuristics.single import SingleHyperHeuristic  
# 导入单一HyperHeuristic类，用于单启发式算法执行
from src.pipeline.hyper_heuristics.perturbation import PerturbationHyperHeuristic 
# 导入PerturbationHyperHeuristic类，用于带扰动的启发式执行
from src.util.util import df_to_str, extract, filter_dict_to_str, parse_text_to_dict 
from src.util.util import load_function, extract_function_with_short_docstring, search_file  
# 从工具模块导入多个工具函数，用于数据处理、提取、加载函数等
from src.util.llm_client.base_llm_client import BaseLLMClient 
# 导入LLM客户端基类，用于与大语言模型交互
class HeuristicEvolver:  # 定义启发式进化器类，用于实现启发式算法的进化过程
    def __init__(  # 类的初始化方法，用于设置初始参数和加载必要组件
        self,
        llm_client: BaseLLMClient,  # LLM客户端实例，用于调用大语言模型
        problem: str,  # 问题类型字符串，指定当前要解决的组合优化问题
        evolution_dir: str,  # 进化数据集目录路径，用于启发式进化的训练数据
        validation_dir: str,  # 验证数据集目录路径，用于评估启发式性能
    ) -> None:  # 初始化方法无返回值
        self.llm_client = llm_client  # 将传入的LLM客户端赋值给实例变量
        self.problem = problem  # 将问题类型赋值给实例变量
        self.evolution_cases = [os.path.join(evolution_dir, f) for f in os.listdir(evolution_dir)]  
        # 生成进化数据集中所有文件的路径列表
        self.validation_cases = [os.path.join(validation_dir, f) for f in os.listdir(validation_dir)] 
        # 生成验证数据集中所有文件的路径列表
        self.get_instance_problem_state = load_function("problem_state.py", problem=self.problem, 
                                                        function_name="get_instance_problem_state") 
        # 加载获取实例问题状态的函数
        self.get_solution_problem_state = load_function("problem_state.py", problem=self.problem, 
                                                        function_name="get_solution_problem_state") 
        # 加载获取解决方案问题状态的函数

        # 加载环境模块
        module = importlib.import_module(f"src.problems.{problem}.env")  
        # 动态导入当前问题对应的环境模块
        globals()["Env"] = getattr(module, "Env") 
        # 将环境类添加到全局变量，方便后续使用

        # 加载组件模块
        if os.path.exists(os.path.join("src", "problems", self.problem, "components.py")): 
            # 检查当前问题是否有自定义组件文件
            module = importlib.import_module(f"src.problems.{self.problem}.components") 
            # 导入自定义组件模块
        else:  # 如果没有自定义组件
            module = importlib.import_module(f"src.problems.base.mdp_components")  
            # 导入基础组件模块
        names_to_import = (name for name in dir(module) if not name.startswith('_')) 
        # 筛选出模块中所有非私有名称
        for name in names_to_import:  # 遍历这些名称
            globals()[name] = getattr(module, name) 
            # 将它们添加到全局变量，方便后续使用
        
        # 准备验证特征数据
        instance_problem_states = []  # 初始化实例问题状态列表
        for data in self.validation_cases:  # 遍历所有验证数据
            global_data = Env(data_name=data).instance_data  
            # 创建环境实例并获取实例数据
            instance_problem_state = {"data_name": data.split(os.sep)[-1]}  
            # 初始化实例问题状态字典，包含数据名称
            instance_problem_state.update(self.get_instance_problem_state(global_data))  
            # 调用函数获取实例问题状态并更新到字典
            instance_problem_states.append(instance_problem_state) 
            # 将字典添加到列表
        self.instance_problem_states_df = pd.DataFrame(instance_problem_states) 
        # 将列表转换为DataFrame，存储为实例变量

    def evolve(  # 定义进化方法，用于执行多轮启发式进化
            self,
            basic_heuristic_file: str,  # 基础启发式算法文件路径
            perturbation_heuristic_file: str,  # 扰动启发式算法文件路径
            perturbation_ratio: float=0.1,  # 扰动比例，默认为0.1
            perturbation_time: int=100,  # 最大扰动尝试次数，默认为100
            filtered_num: int=3,  # 每轮进化后保留的顶级启发式数量，默认为3
            evolution_round: int=3,  # 进化总轮数，默认为3
            max_refinement_round: int=5,  # 每个启发式的最大优化轮次，默认为5
            smoke_test: bool=True,  # 是否进行快速测试，默认为True
        ) -> None:  # 方法返回过滤后的启发式基准列表

        # 准备当前进化过程中其他启发式的描述信息
        heuristic_dir = os.path.dirname(basic_heuristic_file)  
        # 获取基础启发式文件所在目录

        heuristic_introduction_docs = "\n".join([  # 生成所有启发式的简介文档
            extract_function_with_short_docstring(open(search_file(heuristic_file, self.problem)).read(), 
                                                  heuristic_file.split(".")[0])  # 提取每个启发式的简短文档字符串
            for heuristic_file in os.listdir(heuristic_dir)  # 遍历目录中的所有启发式文件
        ])

        total_heuristic_benchmarks = [(basic_heuristic_file, 0)]  
        # 初始化总启发式基准列表，包含基础启发式及其初始改进值
        for _ in range(evolution_round):  # 循环执行指定轮数的进化
            # 筛选出表现最好的启发式
            filtered_heuristic_benchmarks = sorted(
                total_heuristic_benchmarks, key=lambda x: x[1], reverse=True)[: filtered_num]  # 按改进值降序排序并取前filtered_num个
            for basic_heuristic_file, _ in filtered_heuristic_benchmarks:  # 遍历每一个筛选出的启发式
                for data_name in self.evolution_cases:  # 遍历每一个进化数据
                    evolved_heuristic_with_improvements = self.evolution_single(  
                        # 调用单轮进化方法，对当前启发式在当前数据上进行进化
                        evolution_data=data_name,  # 进化数据路径
                        basic_heuristic_file=basic_heuristic_file,  # 基础启发式文件
                        perturbation_heuristic_file=perturbation_heuristic_file,  # 扰动启发式文件
                        all_heuristic_docs=heuristic_introduction_docs,  # 所有启发式的文档
                        perturbation_ratio=perturbation_ratio,  # 扰动比例
                        perturbation_time=perturbation_time,  # 扰动尝试次数
                        max_refinement_round=max_refinement_round,  # 最大优化轮次
                        smoke_test=smoke_test  # 是否快速测试
                    )
                    total_heuristic_benchmarks.extend(evolved_heuristic_with_improvements) 
                    # 将进化得到的启发式及其改进值添加到总列表
        return filtered_heuristic_benchmarks  # 返回最后筛选出的启发式基准列表

    def evolution_single(  # 定义单轮进化方法，对单个启发式在单个数据上进行进化
            self,
            evolution_data: str,  # 进化数据路径
            basic_heuristic_file: str,  # 基础启发式文件路径
            perturbation_heuristic_file: str,  # 扰动启发式文件路径
            all_heuristic_docs: str,  # 所有启发式的文档
            perturbation_ratio: float=0.1,  # 扰动比例
            perturbation_time: int=100,  # 扰动尝试次数
            max_refinement_round: int=5,  # 最大优化轮次
            smoke_test: bool=True  # 是否快速测试
    ) -> list[tuple[str, list[float]]]:  # 返回进化后的启发式文件路径及其改进值的列表
        try:  # 尝试执行以下代码，捕获可能的异常
            env = Env(data_name=evolution_data)  # 创建环境实例，加载进化数据
            basic_heuristic_name = basic_heuristic_file.split(os.sep)[-1].split(".")[0] 
            # 从文件路径中提取基础启发式名称
            output_dir = os.path.join("output", self.problem, "evolution_result",
                                      f"{basic_heuristic_name}.evolution", env.data_ref_name)  # 定义输出目录路径
            self.llm_client.reset(output_dir)  # 重置LLM客户端，设置输出目录

            # 通过扰动生成更好的解决方案
            negative_result, positive_result = self.perturbation(  
                # 调用扰动方法，生成负样本（基础启发式结果）和正样本（更优结果）
                env,  # 环境实例
                basic_heuristic_file,  # 基础启发式文件
                perturbation_heuristic_file,  # 扰动启发式文件
                output_dir,  # 输出目录
                perturbation_ratio,  # 扰动比例
                perturbation_time,  # 扰动尝试次数
            )

            refined_heuristic_benchmarks = []  # 初始化优化后的启发式基准列表
            if positive_result:  # 如果成功生成了更优的正样本结果
                print(f"Evolution {basic_heuristic_name} on {evolution_data}")  # 打印进化信息

                prompt_dict = self.llm_client.load_background(self.problem, 
                                                              "background_with_code") 
                # 加载问题背景信息到提示字典
                prompt_dict["all_heuristic_docs"] = all_heuristic_docs
                # 将所有启发式文档添加到提示字典
                self.load_function_code(basic_heuristic_file, prompt_dict) 
                # 加载基础启发式的代码到提示字典

                # 识别瓶颈操作
                bottlenecks = self.identity_bottlenecks(  
                    # 调用识别瓶颈方法，找出导致基础启发式性能不佳的操作
                    prompt_dict=prompt_dict,  # 提示字典
                    env=env,  # 环境实例
                    positive_result=positive_result,  # 正样本结果
                    negative_result=negative_result  # 负样本结果
                )

                for bottleneck_index, (bottleneck_operation_id, 
                                       proposed_operation, reason) in enumerate(bottlenecks):  # 遍历每个瓶颈
                    # 提出改进建议并生成进化后的启发式
                    basic_heuristic_result = self.validation(self.validation_cases,
                                                             basic_heuristic_file)  # 验证基础启发式在验证集上的表现
                    suggestion_name = f"suggestion_{bottleneck_index}"  # 定义建议名称
                    suggested_heuristic_file, suggestion, suggested_result = self.raise_suggestion( 
                        # 调用提出建议方法，生成改进后的启发式
                        prompt_dict=prompt_dict,  # 提示字典
                        env=env,  # 环境实例
                        bottleneck_operation_id=bottleneck_operation_id,  # 瓶颈操作ID
                        proposed_operation=proposed_operation,  # 建议的操作
                        reason=reason,  # 建议的理由
                        suggestion_name=suggestion_name,  # 建议名称
                        smoke_test=smoke_test  # 是否快速测试
                    )
                    if suggested_heuristic_file:  # 如果成功生成了改进的启发式文件
                        output_heuristic_name = suggested_heuristic_file.split(os.sep)[-1].split(".")[0] 
                        # 提取输出启发式名称
                        self.llm_client.dump(f"{basic_heuristic_name}_to_{output_heuristic_name}")  
                        # 保存LLM交互记录

                        suggested_improvement = sum(self.get_improvement(
                                            env, basic_heuristic_result, suggested_result)) / len(basic_heuristic_result) 
                        # 计算平均改进值
                        print(f"Improvement for {suggested_heuristic_file}: {suggested_improvement}")  # 打印改进信息
                        refined_heuristic_benchmarks.append([suggested_heuristic_file, suggested_improvement]) 
                        # 将改进的启发式及其改进值添加到列表
                        # 进一步微调进化后的启发式
                        previous_heuristic_name = basic_heuristic_name  # 记录上一个启发式名称
                        previous_heuristic_result = basic_heuristic_result  # 记录上一个启发式结果
                        last_heuristic_name = suggested_heuristic_file.split(os.sep)[-1].split(".")[0]  
                        # 记录当前启发式名称
                        last_heuristic_result = suggested_result  # 记录当前启发式结果
                        last_suggestion = suggestion  # 记录上一个建议
                        for refine_index in range(max_refinement_round):  # 循环执行最大优化轮次
                            suggestion_name = f"suggestion_{bottleneck_index}_refine_{refine_index}"  
                            # 定义优化建议名称
                            refined_heuristic_file, suggestion, refined_result = self.refine_heuristic(  
                                # 调用优化启发式方法，进一步改进启发式
                                prompt_dict=prompt_dict,  # 提示字典
                                env=env,  # 环境实例
                                basic_heuristic_name=basic_heuristic_name,  # 基础启发式名称
                                basic_heuristic_result=basic_heuristic_result,  # 基础启发式结果
                                previous_heuristic_name=previous_heuristic_name,  # 上一个启发式名称
                                previous_heuristic_result=previous_heuristic_result,  # 上一个启发式结果
                                last_heuristic_name=last_heuristic_name,  # 当前启发式名称
                                last_heuristic_result=last_heuristic_result,  # 当前启发式结果
                                last_suggestion=last_suggestion,  # 上一个建议
                                suggestion_name=suggestion_name,  # 建议名称
                                smoke_test=smoke_test  # 是否快速测试
                            )
                            if None in refined_result:  # 如果优化结果中有None（表示出错）
                                print("Error and skip")  # 打印错误信息并跳过
                                continue  # 继续下一轮优化
                            if refined_heuristic_file:  # 如果成功生成了优化后的启发式文件
                                output_heuristic_name = refined_heuristic_file.split(os.sep)[-1].split(".")[0]  
                                # 提取输出启发式名称
                                self.llm_client.dump(f"{last_heuristic_name}_to_{output_heuristic_name}") 
                                # 保存LLM交互记录
                                refined_improvement = sum(self.get_improvement(
                                                    env, basic_heuristic_result, refined_result)) / len(basic_heuristic_result)  # 计算平均改进值
                                print(f"Improvement for {refined_heuristic_file}: {refined_improvement}") 
                                # 打印改进信息
                                refined_heuristic_benchmarks.append([refined_heuristic_file, refined_improvement]) 
                                # 将优化的启发式及其改进值添加到列表
                                if suggestion is None:  # 如果没有新的建议
                                    break  # 跳出优化循环
                                previous_heuristic_name = last_heuristic_name  # 更新上一个启发式名称
                                previous_heuristic_result = last_heuristic_result  # 更新上一个启发式结果
                                last_suggestion = suggestion  # 更新上一个建议
                                last_heuristic_name = refined_heuristic_file.split(os.sep)[-1].split(".")[0]  
                                # 更新当前启发式名称
                                last_heuristic_result = refined_result  # 更新当前启发式结果
        except Exception as e:  # 捕获异常
            trace_string = traceback.format_exc()  # 获取异常堆栈信息
            print(trace_string)  # 打印异常信息

        return refined_heuristic_benchmarks  # 返回优化后的启发式基准列表

    def perturbation(  # 定义扰动方法，用于生成基础启发式的结果（负样本）和更优的结果（正样本）
            self,
            env: BaseEnv,  # 环境实例
            basic_heuristic_file: str,  # 基础启发式文件路径
            perturbation_heuristic_file: str,  # 扰动启发式文件路径
            output_dir: str,  # 输出目录路径
            perturbation_ratio: float=0.1,  # 扰动比例
            perturbation_time: int=100,  # 最大扰动尝试次数
        ) -> tuple[bool, str, str]:  # 返回负样本结果和正样本结果
        env.reset(output_dir)  # 重置环境，设置输出目录

        # 用基础启发式生成负样本结果
        hyper_heuristic = SingleHyperHeuristic(basic_heuristic_file, problem=self.problem)  
        # 创建单启发式执行器实例
        hyper_heuristic.run(env)  # 运行基础启发式
        negative_result = env.dump_result(dump_records=["operation_id", "operator"], result_file="negative_solution.txt")
        # 保存负样本结果并返回
        negative_value = env.key_value  # 记录负样本的关键值（如成本、收益等）

        # 用扰动启发式生成正样本结果
        positive_result = None  # 初始化正样本结果为None
        for _ in range(perturbation_time):  # 循环尝试指定次数的扰动
            env.reset(output_dir)  # 重置环境
            hyper_heuristic = PerturbationHyperHeuristic(basic_heuristic_file, 
                                                         perturbation_heuristic_file,
                                                         self.problem, perturbation_ratio)  # 创建带扰动的启发式执行器实例
            hyper_heuristic.run(env)  # 运行扰动启发式
            if env.compare(env.key_value, negative_value) > 0:  # 如果当前结果优于负样本结果
                positive_result = env.dump_result(dump_records=["operation_id", "operator"], result_file="positive_solution.txt")  # 保存正样本结果并返回
                break  # 跳出循环，无需继续尝试
        return negative_result, positive_result  # 返回负样本结果和正样本结果

    def load_function_code(self, heuristic_file: str, prompt_dict: dict) -> str:  # 定义加载函数代码的方法，用于将启发式代码加载到提示字典
        heuristic_file = search_file(heuristic_file, problem=self.problem)  # 查找启发式文件的实际路径
        function_name = heuristic_file.split(os.sep)[-1].split(".")[0]  # 从文件路径中提取函数名称
        function_code = open(heuristic_file).read()  # 读取函数代码
        heuristic_name = function_name[:-5]  # 提取启发式名称（假设函数名末尾有5个字符的后缀）
        prompt_dict["function_name"] = function_name  # 将函数名添加到提示字典
        prompt_dict["function_code"] = function_code  # 将函数代码添加到提示字典
        prompt_dict["heuristic_name"] = heuristic_name  # 将启发式名称添加到提示字典
        return function_code  # 返回函数代码

    def identity_bottlenecks(  # 定义识别瓶颈的方法，用于找出基础启发式中导致性能不佳的操作
            self,
            prompt_dict: dict,  # 提示字典
            env: BaseEnv,  # 环境实例
            positive_result: str,  # 正样本结果
            negative_result: str,  # 负样本结果
        ) -> list[list[int, str, str, str]]:  # 返回瓶颈操作列表，每个元素包含操作ID、建议操作和理由
        env.reset()  # 重置环境

        # 加载数据到提示字典
        prompt_dict["instance_data"] = filter_dict_to_str(env.instance_data)  # 将实例数据过滤并转换为字符串后添加到提示字典
        prompt_dict["instance_problem_state"] = filter_dict_to_str(self.get_instance_problem_state(env.instance_data))  # 将实例问题状态过滤并转换为字符串后添加到提示字典

        # 加载解决方案信息到提示字典
        positive_result = parse_text_to_dict(positive_result)  # 将正样本结果解析为字典
        negative_result = parse_text_to_dict(negative_result)  # 将负样本结果解析为字典
        prompt_dict["positive_solution"] = positive_result["current_solution"]  # 将正样本的当前解决方案添加到提示字典
        prompt_dict["negative_solution"] = negative_result["current_solution"]  # 将负样本的当前解决方案添加到提示字典
        prompt_dict["positive_result"] = positive_result[env.key_item]  # 将正样本的关键结果添加到提示字典
        prompt_dict["negative_result"] = negative_result[env.key_item]  # 将负样本的关键结果添加到提示字典
        prompt_dict["positive_trajectory"] = positive_result["trajectory"]  # 将正样本的执行轨迹添加到提示字典
        prompt_dict["negative_trajectory"] = negative_result["trajectory"]  # 将负样本的执行轨迹添加到提示字典

        # 调用LLM识别瓶颈操作
        self.llm_client.load("identify_bottleneck", prompt_dict)  # 加载识别瓶颈的提示模板和参数
        response = self.llm_client.chat()  # 与LLM交互，获取响应
        bottleneck_operation_strs = extract(response, key="bottleneck_operations", sep="\n")  # 从响应中提取瓶颈操作字符串列表
        self.llm_client.dump("bottleneck_operations")  # 保存LLM交互记录

        bottlenecks = []  # 初始化瓶颈列表
        for bottleneck_operation_str in bottleneck_operation_strs:  # 遍历每个瓶颈操作字符串
            # 解析瓶颈操作信息
            bottleneck_operation_id, proposed_operation, reason = bottleneck_operation_str.split(";")  # 按分号分割字符串，获取操作ID、建议操作和理由
            bottleneck_operation_id = int(re.search(r'\d+', bottleneck_operation_id).group())  # 从操作ID字符串中提取数字并转换为整数
            bottlenecks.append([bottleneck_operation_id, proposed_operation, reason])  # 将瓶颈信息添加到列表

        return bottlenecks  # 返回瓶颈列表

    def raise_suggestion(  # 定义提出建议的方法，用于根据瓶颈生成改进建议并生成新的启发式
            self,
            prompt_dict: dict,  # 提示字典
            env: BaseEnv,  # 环境实例
            bottleneck_operation_id: int,  # 瓶颈操作ID
            proposed_operation: str,  # 建议的操作
            reason: str,  # 建议的理由
            suggestion_name: str,  # 建议名称
            smoke_test: bool,  # 是否快速测试
    ) -> tuple[str, str]:  # 返回新的启发式文件路径、建议和其验证结果
        env.reset()  # 重置环境
        self.llm_client.load_chat("bottleneck_operations")  # 加载之前识别瓶颈的聊天记录
        prompt_dict["bottleneck_operation_id"] = bottleneck_operation_id  # 将瓶颈操作ID添加到提示字典
        prompt_dict["proposed_operation"] = proposed_operation  # 将建议的操作添加到提示字典
        prompt_dict["reason"] = reason  # 将建议的理由添加到提示字典
        negative_trajectory_df = pd.read_csv(StringIO(prompt_dict["negative_trajectory"]), sep="\t")  # 将负样本轨迹字符串转换为DataFrame
        bottleneck_operation = list(negative_trajectory_df[negative_trajectory_df["operation_id"] == bottleneck_operation_id]["operator"])[0]  # 获取瓶颈操作的具体内容

        for previous_operation in negative_trajectory_df[negative_trajectory_df["operation_id"] < bottleneck_operation_id]["operator"]:  # 遍历瓶颈操作之前的所有操作
            env.run_operator(eval(previous_operation))  # 在环境中执行这些操作，重现瓶颈前的状态
        prompt_dict["bottleneck_operation"] = bottleneck_operation  # 将瓶颈操作添加到提示字典
        prompt_dict["solution_before_bottleneck"] = str(env.current_solution)  # 将瓶颈前的解决方案添加到提示字典
        prompt_dict["solution_problem_state_before_bottleneck"] = filter_dict_to_str(self.get_solution_problem_state(env.instance_data, env.current_solution))  # 将瓶颈前的解决方案问题状态添加到提示字典

        # 调用LLM获取改进建议
        self.llm_client.load("extract_suggestion", prompt_dict)  # 加载提取建议的提示模板和参数
        response = self.llm_client.chat()  # 与LLM交互，获取响应
        suggestion = extract(response, key="suggestion")  # 从响应中提取建议
        if suggestion:  # 如果获取到建议
            self.llm_client.dump(suggestion_name)  # 保存LLM交互记录
            # 根据建议生成新的启发式代码
            heuristic_name = prompt_dict["heuristic_name"]  # 获取启发式名称
            origin_function_name = prompt_dict["function_name"]  # 获取原始函数名称
            prompt_dict["suggestion"] = suggestion  # 将建议添加到提示字典
            description = f"Now, based on these suggestions:\n{suggestion}\nUpdate the {origin_function_name}."  # 生成新启发式的描述
            env_summarize = prompt_dict["env_summarize"]  # 获取环境摘要信息
            output_heuristic_file = HeuristicGenerator(self.llm_client, self.problem).generate(heuristic_name, description, env_summarize, smoke_test)  # 调用启发式生成器生成新的启发式文件
            if output_heuristic_file:  # 如果成功生成新的启发式文件
                suggested_result = self.validation(self.validation_cases, output_heuristic_file)  # 在验证集上验证新启发式
                return output_heuristic_file, suggestion, suggested_result  # 返回新启发式文件路径、建议和验证结果
        return None, None, None  # 如果没有生成有效的新启发式，返回None

    def refine_heuristic(  # 定义优化启发式的方法，用于进一步改进已生成的启发式
            self,
            prompt_dict: dict,  # 提示字典
            env: BaseEnv,  # 环境实例
            basic_heuristic_name: str,  # 基础启发式名称
            basic_heuristic_result: list[float],  # 基础启发式结果
            previous_heuristic_name: str,  # 上一个启发式名称
            previous_heuristic_result: list[float],  # 上一个启发式结果
            last_heuristic_name: str,  # 当前启发式名称
            last_heuristic_result: list[float],  # 当前启发式结果
            last_suggestion: str,  # 上一个建议
            suggestion_name: str,  # 建议名称
            smoke_test: bool=True,  # 是否快速测试
    ) -> tuple[str, str, float]:  # 返回优化后的启发式文件路径、建议和其验证结果
        # 生成基准对比表格
        benchmark_df = self.instance_problem_states_df.copy()  # 复制实例问题状态DataFrame作为基准表格
        benchmark_df[basic_heuristic_name] = basic_heuristic_result  # 添加基础启发式结果列
        previous_improvement = self.get_improvement(env, basic_heuristic_result, previous_heuristic_result)  # 计算上一个启发式相对基础启发式的改进值
        benchmark_df[previous_heuristic_name] = [f"{previous_heuristic_result[index]}({previous_improvement[index]})" for index in range(len(basic_heuristic_result))]  # 添加上一个启发式结果列，包含改进值
        last_improvement = self.get_improvement(env, basic_heuristic_result, last_heuristic_result)  # 计算当前启发式相对基础启发式的改进值
        benchmark_df[last_heuristic_name] = [f"{last_heuristic_result[index]}({last_improvement[index]})" for index in range(len(basic_heuristic_result))]  # 添加当前启发式结果列，包含改进值
        # 比较进化后的启发式在验证数据上的结果
        problem_state_num = self.instance_problem_states_df.shape[-1]  # 获取问题状态的数量
        if env.compare(1, 2) > 0:  # 根据环境的比较方法判断优化目标是更小还是更大
            compare = "lower"  # 如果1比2好，说明目标是更小的值
        else:
            compare = "higher"  # 否则目标是更大的值
        prompt_dict["problem_state_num"] = problem_state_num  # 将问题状态数量添加到提示字典
        prompt_dict["basic_heuristic_name"] = basic_heuristic_name  # 将基础启发式名称添加到提示字典
        prompt_dict["previous_heuristic_name"] = previous_heuristic_name  # 将上一个启发式名称添加到提示字典
        prompt_dict["last_heuristic_name"] = last_heuristic_name  # 将当前启发式名称添加到提示字典
        prompt_dict["last_suggestion"] = last_suggestion  # 将上一个建议添加到提示字典
        prompt_dict["compare"] = compare  # 将优化目标（更小/更大）添加到提示字典
        prompt_dict["benchmark_result"] = df_to_str(benchmark_df)  # 将基准表格转换为字符串添加到提示字典

        self.llm_client.load("refinement", prompt_dict)  # 加载优化启发式的提示模板和参数
        response = self.llm_client.chat()  # 与LLM交互，获取响应
        analysis_results = extract(response, key="refinement", sep="\n")  # 从响应中提取分析结果
        self.llm_client.dump(suggestion_name)  # 保存LLM交互记录
        suggestion = None  # 初始化建议为None
        for analysis_result in analysis_results:  # 遍历分析结果
            if "code adjustment suggestion" in analysis_result:  # 如果找到代码调整建议
                suggestion = analysis_result.split(":")[-1]  # 提取建议内容

        if suggestion:  # 如果获取到建议
            # 根据建议生成新的启发式代码
            heuristic_name = prompt_dict["heuristic_name"]  # 获取启发式名称
            prompt_dict["suggestion"] = suggestion  # 将建议添加到提示字典
            description = f"Now, based on these suggestions:\n{suggestion}\nUpdate the {last_heuristic_name}."  # 生成新启发式的描述
            env_summarize = prompt_dict["env_summarize"]  # 获取环境摘要信息
            output_heuristic_file = HeuristicGenerator(self.llm_client, self.problem).generate(heuristic_name, description, env_summarize, smoke_test, reminder=False)  # 调用启发式生成器生成新的启发式文件，不使用提醒
            output_heuristic_name = output_heuristic_file.split(os.sep)[-1].split(".")[0]  # 提取输出启发式名称
            self.llm_client.dump(f"{previous_heuristic_name}_to_{output_heuristic_name}")  # 保存LLM交互记录
            if output_heuristic_file:  # 如果成功生成新的启发式文件
                suggested_heuristic_result = self.validation(self.validation_cases, output_heuristic_file)  # 在验证集上验证新启发式
                return output_heuristic_file, suggestion, suggested_heuristic_result  # 返回新启发式文件路径、建议和验证结果
        return None, None, None  # 如果没有生成有效的新启发式，返回None

    def validation(  # 定义验证方法，用于在验证集上评估启发式的性能
            self,
            validation_cases: list[str],  # 验证数据集路径列表
            heuristic_file: str  # 要验证的启发式文件路径
        ) -> list[float]:  # 返回每个验证数据上的结果值列表
        validation_results = []  # 初始化验证结果列表
        heuristic_name = heuristic_file.split(os.sep)[-1].split(".py")[0]  # 从文件路径中提取启发式名称
        for data_name in validation_cases:  # 遍历每个验证数据
            env = Env(data_name=data_name)  # 创建环境实例，加载验证数据
            env.reset(heuristic_name)  # 重置环境，设置启发式名称
            hyper_heuristic = SingleHyperHeuristic(heuristic_file, problem=self.problem)  # 创建单启发式执行器实例