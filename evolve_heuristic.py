import pandas as pd  # 导入pandas库，用于数据处理和分析
import traceback  # 导入traceback库，用于捕获和处理异常堆栈信息
from io import StringIO  # 从io模块导入StringIO，用于在内存中处理字符串形式的文件
from src.problems.base.env import BaseEnv  # 从基础环境模块导入BaseEnv基类
from src.pipeline.heuristic_generator import HeuristicGenerator  # 从启发式生成器模块导入HeuristicGenerator类
from src.pipeline.hyper_heuristics.single import SingleHyperHeuristic  # 从单启发式超启发式模块导入SingleHyperHeuristic类
from src.pipeline.hyper_heuristics.perturbation import PerturbationHyperHeuristic  # 从扰动超启发式模块导入PerturbationHyperHeuristic类
from src.util.util import df_to_str, extract, filter_dict_to_str, parse_text_to_dict, load_function, extract_function_with_short_docstring, search_file  # 从工具模块导入多种工具函数
from src.util.llm_client.base_llm_client import BaseLLMClient  # 从基础LLM客户端模块导入BaseLLMClient基类
class HeuristicEvolver:  # 定义HeuristicEvolver类，用于启发式算法的进化优化
    def __init__(  # 初始化方法：接收LLM客户端、问题名称、进化案例目录、验证案例目录，初始化相关属性
        self,
        llm_client: BaseLLMClient,
        problem: str,
        evolution_dir: str,
        validation_dir: str,
    ) -> None:
        self.llm_client = llm_client  # 保存LLM客户端实例
        self.problem = problem  # 保存问题名称
        self.evolution_cases = [os.path.join(evolution_dir, f) for f in os.listdir(evolution_dir)]  # 获取所有进化案例文件路径
        self.validation_cases = [os.path.join(validation_dir, f) for f in os.listdir(validation_dir)]  # 获取所有验证案例文件路径
        self.get_instance_problem_state = load_function("problem_state.py", problem=self.problem, function_name="get_instance_problem_state")  # 加载获取实例问题状态的函数
        self.get_solution_problem_state = load_function("problem_state.py", problem=self.problem, function_name="get_solution_problem_state")  # 加载获取解决方案问题状态的函数

        # 加载环境模块
        module = importlib.import_module(f"src.problems.{problem}.env")  # 动态导入当前问题的环境模块
        globals()["Env"] = getattr(module, "Env")  # 将Env类添加到全局变量，方便后续使用

        # 加载组件模块
        if os.path.exists(os.path.join("src", "problems", self.problem, "components.py")):  # 若当前问题有自定义组件文件
            module = importlib.import_module(f"src.problems.{self.problem}.components")  # 导入自定义组件模块
        else:  # 否则
            module = importlib.import_module(f"src.problems.base.mdp_components")  # 导入基础组件模块
        names_to_import = (name for name in dir(module) if not name.startswith('_'))  # 获取模块中所有非私有属性/类名
        for name in names_to_import:  # 遍历这些名称
            globals()[name] = getattr(module, name)  # 将它们添加到全局变量，方便后续使用
        
        # 准备验证特征数据
        instance_problem_states = []  # 初始化实例问题状态列表
        for data in self.validation_cases:  # 遍历所有验证案例
            global_data = Env(data_name=data).instance_data  # 获取环境的实例数据
            instance_problem_state = {"data_name": data.split(os.sep)[-1]}  # 创建包含数据名称的字典
            instance_problem_state.update(self.get_instance_problem_state(global_data))  # 补充实例问题状态数据
            instance_problem_states.append(instance_problem_state)  # 添加到列表
        self.instance_problem_states_df = pd.DataFrame(instance_problem_states)  # 转换为DataFrame，便于后续分析

    def evolve(  # 进化主方法：对基础启发式算法进行多轮进化，返回优化后的启发式算法
            self,
            basic_heuristic_file: str,
            perturbation_heuristic_file: str,
            perturbation_ratio: float=0.1,
            perturbation_time: int=100,
            filtered_num: int=3,
            evolution_round: int=3,
            max_refinement_round: int=5,
            smoke_test: bool=True,
        ) -> None:

        # 准备当前进化所需的其他启发式算法描述
        heuristic_dir = os.path.dirname(basic_heuristic_file)  # 获取基础启发式算法文件所在目录

        heuristic_introduction_docs = "\n".join([  # 拼接所有启发式算法的简介文档
            extract_function_with_short_docstring(open(search_file(heuristic_file, self.problem)).read(), heuristic_file.split(".")[0])
            for heuristic_file in os.listdir(heuristic_dir)
        ])

        total_heuristic_benchmarks = [(basic_heuristic_file, 0)]  # 初始化启发式算法基准列表，包含基础启发式算法
        for _ in range(evolution_round):  # 进行指定轮数的进化
            # 筛选表现最佳的启发式算法
            filtered_heuristic_benchmarks = sorted(total_heuristic_benchmarks, key=lambda x: x[1], reverse=True)[: filtered_num]  # 按性能排序并取前filtered_num个
            for basic_heuristic_file, _ in filtered_heuristic_benchmarks:  # 遍历筛选后的启发式算法
                for data_name in self.evolution_cases:  # 遍历所有进化案例
                    evolved_heuristic_with_improvements = self.evolution_single(  # 对单个案例进行进化
                        evolution_data=data_name,
                        basic_heuristic_file=basic_heuristic_file,
                        perturbation_heuristic_file=perturbation_heuristic_file,
                        all_heuristic_docs=heuristic_introduction_docs,
                        perturbation_ratio=perturbation_ratio,
                        perturbation_time=perturbation_time,
                        max_refinement_round=max_refinement_round,
                        smoke_test=smoke_test
                    )
                    total_heuristic_benchmarks.extend(evolved_heuristic_with_improvements)  # 将进化得到的启发式算法添加到基准列表
        return filtered_heuristic_benchmarks  # 返回筛选后的启发式算法

    def evolution_single(  # 单个案例的进化方法：针对特定数据案例进化启发式算法，返回改进后的启发式算法及改进幅度
            self,
            evolution_data: str,
            basic_heuristic_file: str,
            perturbation_heuristic_file: str,
            all_heuristic_docs: str,
            perturbation_ratio: float=0.1,
            perturbation_time: int=100,
            max_refinement_round: int=5,
            smoke_test: bool=True
    ) -> list[tuple[str, list[float]]]:
        try:  # 捕获可能的异常
            env = Env(data_name=evolution_data)  # 创建环境实例
            basic_heuristic_name = basic_heuristic_file.split(os.sep)[-1].split(".")[0]  # 提取基础启发式算法名称
            output_dir = os.path.join("output", self.problem, "evolution_result", f"{basic_heuristic_name}.evolution", env.data_ref_name)  # 定义输出目录
            self.llm_client.reset(output_dir)  # 重置LLM客户端，设置输出目录

            # 通过扰动获取更优解
            negative_result, positive_result = self.perturbation(  # 调用扰动方法，获取基础解（negative）和更优解（positive）
                env,
                basic_heuristic_file,
                perturbation_heuristic_file,
                output_dir,
                perturbation_ratio,
                perturbation_time,
            )

            refined_heuristic_benchmarks = []  # 初始化改进后的启发式算法基准列表
            if positive_result:  # 若存在更优解
                print(f"Evolution {basic_heuristic_name} on {evolution_data}")  # 打印进化信息

                prompt_dict = self.llm_client.load_background(self.problem, "background_with_code")  # 加载问题背景到提示词字典
                prompt_dict["all_heuristic_docs"] = all_heuristic_docs  # 添加所有启发式算法文档到提示词字典
                self.load_function_code(basic_heuristic_file, prompt_dict)  # 加载基础启发式算法代码到提示词字典

                # 识别瓶颈操作
                bottlenecks = self.identity_bottlenecks(  # 调用方法识别瓶颈操作
                    prompt_dict=prompt_dict,
                    env=env,
                    positive_result=positive_result,
                    negative_result=negative_result
                )

                for bottleneck_index, (bottleneck_operation_id, proposed_operation, reason) in enumerate(bottlenecks):  # 遍历每个瓶颈操作
                    # 提出改进建议并生成进化后的启发式算法
                    basic_heuristic_result = self.validation(self.validation_cases, basic_heuristic_file)  # 验证基础启发式算法在验证案例上的表现
                    suggestion_name = f"suggestion_{bottleneck_index}"  # 定义建议名称
                    suggested_heuristic_file, suggestion, suggested_result = self.raise_suggestion(  # 调用方法提出建议并生成新的启发式算法
                        prompt_dict=prompt_dict,
                        env=env,
                        bottleneck_operation_id=bottleneck_operation_id,
                        proposed_operation=proposed_operation,
                        reason=reason,
                        suggestion_name=suggestion_name,
                        smoke_test=smoke_test
                    )
                    if suggested_heuristic_file:  # 若生成了新的启发式算法文件
                        output_heuristic_name = suggested_heuristic_file.split(os.sep)[-1].split(".")[0]  # 提取新启发式算法名称
                        self.llm_client.dump(f"{basic_heuristic_name}_to_{output_heuristic_name}")  # 保存LLM交互记录

                        suggested_improvement = sum(self.get_improvement(env, basic_heuristic_result, suggested_result)) / len(basic_heuristic_result)  # 计算平均改进幅度
                        print(f"Improvement for {suggested_heuristic_file}: {suggested_improvement}")  # 打印改进幅度
                        refined_heuristic_benchmarks.append([suggested_heuristic_file, suggested_improvement])  # 添加到基准列表
                        # 微调进化后的启发式算法
                        previous_heuristic_name = basic_heuristic_name  # 记录上一个启发式算法名称
                        previous_heuristic_result = basic_heuristic_result  # 记录上一个启发式算法结果
                        last_heuristic_name = suggested_heuristic_file.split(os.sep)[-1].split(".")[0]  # 记录当前启发式算法名称
                        last_heuristic_result = suggested_result  # 记录当前启发式算法结果
                        last_suggestion = suggestion  # 记录上一个建议
                        for refine_index in range(max_refinement_round):  # 进行指定轮数的微调
                            suggestion_name = f"suggestion_{bottleneck_index}_refine_{refine_index}"  # 定义微调建议名称
                            refined_heuristic_file, suggestion, refined_result = self.refine_heuristic(  # 调用方法微调启发式算法
                                prompt_dict=prompt_dict,
                                env=env,
                                basic_heuristic_name=basic_heuristic_name,
                                basic_heuristic_result=basic_heuristic_result,
                                previous_heuristic_name=previous_heuristic_name,
                                previous_heuristic_result=previous_heuristic_result,
                                last_heuristic_name=last_heuristic_name,
                                last_heuristic_result=last_heuristic_result,
                                last_suggestion=last_suggestion,
                                suggestion_name=suggestion_name,
                                smoke_test=smoke_test
                            )
                            if None in refined_result:  # 若结果中存在None（错误）
                                print("Error and skip")  # 打印信息并跳过
                                continue
                            if refined_heuristic_file:  # 若生成了微调后的启发式算法文件
                                output_heuristic_name = refined_heuristic_file.split(os.sep)[-1].split(".")[0]  # 提取微调后启发式算法名称
                                self.llm_client.dump(f"{last_heuristic_name}_to_{output_heuristic_name}")  # 保存LLM交互记录
                                refined_improvement = sum(self.get_improvement(env, basic_heuristic_result, refined_result)) / len(basic_heuristic_result)  # 计算平均改进幅度
                                print(f"Improvement for {refined_heuristic_file}: {refined_improvement}")  # 打印改进幅度
                                refined_heuristic_benchmarks.append([refined_heuristic_file, refined_improvement])  # 添加到基准列表
                                if suggestion is None:  # 若没有新的建议
                                    break  # 停止微调
                                previous_heuristic_name = last_heuristic_name  # 更新上一个启发式算法名称
                                previous_heuristic_result = last_heuristic_result  # 更新上一个启发式算法结果
                                last_suggestion = suggestion  # 更新上一个建议
                                last_heuristic_name = refined_heuristic_file.split(os.sep)[-1].split(".")[0]  # 更新当前启发式算法名称
                                last_heuristic_result = refined_result  # 更新当前启发式算法结果
        except Exception as e:  # 捕获异常
            trace_string = traceback.format_exc()  # 获取异常堆栈信息
            print(trace_string)  # 打印异常信息

        return refined_heuristic_benchmarks  # 返回改进后的启发式算法基准列表

    def perturbation(  # 扰动方法：通过扰动基础启发式算法获取更优解，返回基础解和更优解
            self,
            env: BaseEnv,
            basic_heuristic_file: str,
            perturbation_heuristic_file: str,
            output_dir: str,
            perturbation_ratio: float=0.1,
            perturbation_time: int=100,
        ) -> tuple[bool, str, str]:
        env.reset(output_dir)  # 重置环境，设置输出目录

        # 用基础启发式算法生成基础解（negative result）
        hyper_heuristic = SingleHyperHeuristic(basic_heuristic_file, problem=self.problem)  # 创建单启发式超启发式实例
        hyper_heuristic.run(env)  # 运行超启发式算法
        negative_result = env.dump_result(dump_records=["operation_id", "operator"], result_file="negative_solution.txt")  # 保存并获取基础解结果
        negative_value = env.key_value  # 记录基础解的关键值（如成本、收益等）

        # 用扰动启发式算法生成更优解（positive result）
        positive_result = None  # 初始化更优解为None
        for _ in range(perturbation_time):  # 尝试指定次数的扰动
            env.reset(output_dir)  # 重置环境
            hyper_heuristic = PerturbationHyperHeuristic(basic_heuristic_file, perturbation_heuristic_file, self.problem, perturbation_ratio)  # 创建扰动超启发式实例
            hyper_heuristic.run(env)  # 运行扰动超启发式算法
            if env.compare(env.key_value, negative_value) > 0:  # 若当前解优于基础解
                positive_result = env.dump_result(dump_records=["operation_id", "operator"], result_file="positive_solution.txt")  # 保存并获取更优解结果
                break  # 找到更优解后停止尝试
        return negative_result, positive_result  # 返回基础解和更优解

    def load_function_code(self, heuristic_file: str, prompt_dict: dict) -> str:  # 加载启发式函数代码到提示词字典的方法
        heuristic_file = search_file(heuristic_file, problem=self.problem)  # 搜索启发式算法文件
        function_name = heuristic_file.split(os.sep)[-1].split(".")[0]  # 提取函数名称
        function_code = open(heuristic_file).read()  # 读取函数代码
        heuristic_name = function_name[:-5]  # 提取启发式算法名称（去除后缀）
        prompt_dict["function_name"] = function_name  # 向提示词字典添加函数名称
        prompt_dict["function_code"] = function_code  # 向提示词字典添加函数代码
        prompt_dict["heuristic_name"] = heuristic_name  # 向提示词字典添加启发式算法名称
        return function_code  # 返回函数代码

    def identity_bottlenecks(  # 识别瓶颈操作的方法：找出导致基础解性能不佳的关键操作
            self,
            prompt_dict: dict,
            env: BaseEnv,
            positive_result: str,
            negative_result: str,
        ) -> list[list[int, str, str, str]]:
        env.reset()  # 重置环境

        # 加载数据到提示词字典
        prompt_dict["instance_data"] = filter_dict_to_str(env.instance_data)  # 添加过滤后的实例数据
        prompt_dict["instance_problem_state"] = filter_dict_to_str(self.get_instance_problem_state(env.instance_data))  # 添加过滤后的实例问题状态

        # 加载解决方案到提示词字典
        positive_result = parse_text_to_dict(positive_result)  # 解析更优解结果为字典
        negative_result = parse_text_to_dict(negative_result)  # 解析基础解结果为字典
        prompt_dict["positive_solution"] = positive_result["current_solution"]  # 添加更优解的当前解决方案
        prompt_dict["negative_solution"] = negative_result["current_solution"]  # 添加基础解的当前解决方案
        prompt_dict["positive_result"] = positive_result[env.key_item]  # 添加更优解的关键值
        prompt_dict["negative_result"] = negative_result[env.key_item]  # 添加基础解的关键值
        prompt_dict["positive_trajectory"] = positive_result["trajectory"]  # 添加更优解的轨迹（操作序列）
        prompt_dict["negative_trajectory"] = negative_result["trajectory"]  # 添加基础解的轨迹（操作序列）

        # 识别瓶颈操作
        self.llm_client.load("identify_bottleneck", prompt_dict)  # 加载识别瓶颈操作的提示词
        response = self.llm_client.chat()  # 与LLM交互获取响应
        bottleneck_operation_strs = extract(response, key="bottleneck_operations", sep="\n")  # 从响应中提取瓶颈操作字符串列表
        self.llm_client.dump("bottleneck_operations")  # 保存LLM交互记录

        bottlenecks = []  # 初始化瓶颈操作列表
        for bottleneck_operation_str in bottleneck_operation_strs:  # 遍历每个瓶颈操作字符串
            # 解析瓶颈操作信息
            bottleneck_operation_id, proposed_operation, reason = bottleneck_operation_str.split(";")  # 分割得到操作ID、建议操作、原因
            bottleneck_operation_id = int(re.search(r'\d+', bottleneck_operation_id).group())  # 提取操作ID的数字部分
            bottlenecks.append([bottleneck_operation_id, proposed_operation, reason])  # 添加到瓶颈操作列表

        return bottlenecks  # 返回瓶颈操作列表

    def raise_suggestion(  # 提出改进建议的方法：针对瓶颈操作生成改进建议并实现新的启发式算法
            self,
            prompt_dict: dict,
            env: BaseEnv,
            bottleneck_operation_id: int,
            proposed_operation: str,
            reason: str,
            suggestion_name: str,
            smoke_test: bool,
    ) -> tuple[str, str]:
        env.reset()  # 重置环境
        self.llm_client.load_chat("bottleneck_operations")  # 加载之前识别瓶颈操作的对话记录
        prompt_dict["bottleneck_operation_id"] = bottleneck_operation_id  # 向提示词字典添加瓶颈操作ID
        prompt_dict["proposed_operation"] = proposed_operation  # 向提示词字典添加建议操作
        prompt_dict["reason"] = reason  # 向提示词字典添加原因
        negative_trajectory_df = pd.read_csv(StringIO(prompt_dict["negative_trajectory"]), sep="\t")  # 解析基础解轨迹为DataFrame
        bottleneck_operation = list(negative_trajectory_df[negative_trajectory_df["operation_id"] == bottleneck_operation_id]["operator"])[0]  # 获取瓶颈操作的具体内容

        for previous_operation in negative_trajectory_df[negative_trajectory_df["operation_id"] < bottleneck_operation_id]["operator"]:  # 执行瓶颈操作之前的所有操作
            env.run_operator(eval(previous_operation))
        prompt_dict["bottleneck_operation"] = bottleneck_operation  # 向提示词字典添加瓶颈操作内容
        prompt_dict["solution_before_bottleneck"] = str(env.current_solution)  # 向提示词字典添加瓶颈操作前的解决方案
        prompt_dict["solution_problem_state_before_bottleneck"] = filter_dict_to_str(self.get_solution_problem_state(env.instance_data, env.current_solution))  # 向提示词字典添加瓶颈操作前的解决方案问题状态

        # 尝试生成改进建议
        self.llm_client.load("extract_suggestion", prompt_dict)  # 加载提取改进建议的提示词
        response = self.llm_client.chat()  # 与LLM交互获取响应
        suggestion = extract(response, key="suggestion")  # 从响应中提取改进建议
        if suggestion:  # 若存在改进建议
            self.llm_client.dump(suggestion_name)  # 保存LLM交互记录
            # 实现新的启发式算法代码
            heuristic_name = prompt_dict["heuristic_name"]  # 获取启发式算法名称
            origin_function_name = prompt_dict["function_name"]  # 获取原始函数名称
            prompt_dict["suggestion"] = suggestion  # 向提示词字典添加改进建议
            description = f"Now, based on these suggestions:\n{suggestion}\nUpdate the {origin_function_name}."  # 构建新启发式算法的描述
            env_summarize = prompt_dict["env_summarize"]  # 获取环境摘要
            output_heuristic_file = HeuristicGenerator(self.llm_client, self.problem).generate(heuristic_name, description, env_summarize, smoke_test)  # 生成新的启发式算法文件
            if output_heuristic_file:  # 若生成成功
                suggested_result = self.validation(self.validation_cases, output_heuristic_file)  # 验证新启发式算法在验证案例上的表现
                return output_heuristic_file, suggestion, suggested_result  # 返回新算法文件、建议和结果
        return None, None, None  # 若未生成建议或算法，返回None

    def refine_heuristic(  # 微调启发式算法的方法：基于基准测试结果进一步优化启发式算法
            self,
            prompt_dict: dict,
            env: BaseEnv,
            basic_heuristic_name: str,
            basic_heuristic_result: list[float],
            previous_heuristic_name: str,
            previous_heuristic_result: list[float],
            last_heuristic_name: str,
            last_heuristic_result: list[float],
            last_suggestion: str,
            suggestion_name: str,
            smoke_test: bool=True,
    ) -> tuple[str, str, float]:
        # 构建基准测试对比表格
        benchmark_df = self.instance_problem_states_df.copy()  # 复制实例问题状态DataFrame
        benchmark_df[basic_heuristic_name] = basic_heuristic_result  # 添加基础启发式算法的结果
        previous_improvement = self.get_improvement(env, basic_heuristic_result, previous_heuristic_result)  # 计算上一个算法相对基础算法的改进幅度
        benchmark_df[previous_heuristic_name] = [f"{previous_heuristic_result[index]}({previous_improvement[index]})" for index in range(len(basic_heuristic_result))]  # 添加上一个算法的结果及改进幅度
        last_improvement = self.get_improvement(env, basic_heuristic_result, last_heuristic_result)  # 计算当前算法相对基础算法的改进幅度
        benchmark_df[last_heuristic_name] = [f"{last_heuristic_result[index]}({last_improvement[index]})" for index in range(len(basic_heuristic_result))]  # 添加当前算法的结果及改进幅度
        # 在验证数据上对比进化后的结果
        problem_state_num = self.instance_problem_states_df.shape[-1]  # 获取问题状态特征的数量
        if env.compare(1, 2) > 0:  # 确定更优的判断标准（值更小还是更大）
            compare = "lower"  # 若1比2好，则值越小越好
        else:
            compare = "higher"  # 否则值越大越好
        prompt_dict["problem_state_num"] = problem_state_num  # 向提示词字典添加问题状态特征数量
        prompt_dict["basic_heuristic_name"] = basic_heuristic_name  # 向提示词字典添加基础启发式算法名称
        prompt_dict["previous_heuristic_name"] = previous_heuristic_name  # 向提示词字典添加上一个启发式算法名称
        prompt_dict["last_heuristic_name"] = last_heuristic_name  # 向提示词字典添加当前启发式算法名称
        prompt_dict["last_suggestion"] = last_suggestion  # 向提示词字典添加上一个建议
        prompt_dict["compare"] = compare  # 向提示词字典添加更优判断标准
        prompt_dict["benchmark_result"] = df_to_str(benchmark_df)  # 向提示词字典添加基准测试结果表格

        self.llm_client.load("refinement", prompt_dict)  # 加载微调建议的提示词
        response = self.llm_client.chat()  # 与LLM交互获取响应
        analysis_results = extract(response, key="refinement", sep="\n")  # 从响应中提取分析结果
        self.llm_client.dump(suggestion_name)  # 保存LLM交互记录
        suggestion = None  # 初始化建议为None
        for analysis_result in analysis_results:  # 遍历分析结果
            if "code adjustment suggestion" in analysis_result:  # 若存在代码调整建议
                suggestion = analysis_result.split(":")[-1]  # 提取建议内容

        if suggestion:  # 若存在建议
            # 实现新的启发式算法代码
            heuristic_name = prompt_dict["heuristic_name"]  # 获取启发式算法名称
            prompt_dict["suggestion"] = suggestion  # 向提示词字典添加建议
            description = f"Now, based on these suggestions:\n{suggestion}\nUpdate the {last_heuristic_name}."  # 构建新启发式算法的描述
            env_summarize = prompt_dict["env_summarize"]  # 获取环境摘要
            output_heuristic_file = HeuristicGenerator(self.llm_client, self.problem).generate(heuristic_name, description, env_summarize, smoke_test, reminder=False)  # 生成微调后的启发式算法文件
            output_heuristic_name = output_heuristic_file.split(os.sep)[-1].split(".")[0]  # 提取微调后算法名称
            self.llm_client.dump(f"{previous_heuristic_name}_to_{output_heuristic_name}")  # 保存LLM交互记录
            if output_heuristic_file:  # 若生成成功
                suggested_heuristic_result = self.validation(self.validation_cases, output_heuristic_file)  # 验证微调后算法在验证案例上的表现
                return output_heuristic_file, suggestion, suggested_heuristic_result  # 返回微调后算法文件、建议和结果
        return None, None, None  # 若未生成建议或算法，返回None

    def validation(  # 验证方法：在验证案例上运行启发式算法并返回结果
            self,
            validation_cases: list[str],
            heuristic_file: str
        ) -> list[float]:
        validation_results = []  # 初始化验证结果列表
        heuristic_name = heuristic_file.split(os.sep)[-1].split(".py")[0]  # 提取启发式算法名称
        for data_name in validation_cases:  # 遍历所有验证案例
            env = Env(data_name=data_name)  # 创建环境实例
            env.reset(heuristic_name)  # 重置环境，设置启发式算法名称
            hyper_heuristic = SingleHyperHeuristic(heuristic_file, problem=self.problem)  # 创建单启发式超启发式实例
            is_complete_valid_solution = hyper_heuristic.run(env)  # 运行超启发式算法，获取是否生成完整解的标志