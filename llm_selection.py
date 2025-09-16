import traceback  # 导入traceback模块，用于捕获和打印异常堆栈信息
from src.problems.base.env import BaseEnv  # 从指定路径导入BaseEnv类，作为环境的基类
from src.util.util import find_closest_match, load_function, extract_function_with_short_docstring, extract, filter_dict_to_str, search_file 
# 从工具模块导入多个工具函数，分别用于查找最匹配字符串、加载函数、提取函数文档、提取内容、过滤字典为字符串、搜索文件
from src.util.llm_client.base_llm_client import BaseLLMClient  # 导入LLM客户端基类，用于与大语言模型交互
from src.util.tts_bon import tts_bon  # 导入tts_bon函数，用于启发式选择的蒙特卡洛树搜索相关逻辑


class LLMSelectionHyperHeuristic:  # 定义LLM选择超启发式类，用于通过LLM动态选择启发式算法解决问题
    def __init__(  # 类的初始化方法，用于设置超启发式的各项参数和加载必要组件
        self,
        llm_client: BaseLLMClient,  # LLM客户端实例，用于与大语言模型交互
        heuristic_pool: list[str],  # 启发式算法池，包含可用启发式的文件名列表
        problem: str,  # 问题名称，用于定位相关文件和配置
        iterations_scale_factor: float=2.0,  # 迭代缩放因子，用于计算最大步骤数
        steps_per_selection: int=5,  # 每次选择后执行的步数
        num_candidate_heuristics: int=3,  # 候选启发式的数量
        rollout_budget: int=10,  # 蒙特卡洛模拟的预算（评估次数）
        problem_state_content_threshold: int=1000,  # 问题状态内容的长度阈值，用于限制输入LLM的内容长度
    ) -> None:
        self.llm_client = llm_client  # 保存LLM客户端实例
        self.problem = problem  # 保存问题名称
        self.heuristic_pool = [heuristic.split(".")[0] for heuristic in heuristic_pool]  # 处理启发式池，去除文件名中的扩展名，仅保留名称
        self.iterations_scale_factor = iterations_scale_factor  # 保存迭代缩放因子
        self.steps_per_selection = steps_per_selection  # 保存每次选择后执行的步数
        self.num_candidate_heuristics = num_candidate_heuristics  # 保存候选启发式数量
        self.rollout_budget = rollout_budget  # 保存蒙特卡洛预算
        self.problem_state_content_threshold = problem_state_content_threshold  # 保存问题状态内容阈值

        self.heuristic_docs = {  # 构建启发式文档字典，键为启发式名称，值为提取的函数简短文档字符串
            heuristic: extract_function_with_short_docstring(open(search_file(heuristic + ".py", problem)).read(), heuristic) 
            for heuristic in self.heuristic_pool}
        self.heuristic_functions = {  # 构建启发式函数字典，键为启发式名称，值为加载的函数对象
            heuristic.split(".")[0]: load_function(heuristic, problem=self.problem)
            for heuristic in self.heuristic_pool}
        self.get_instance_problem_state = load_function("problem_state.py", problem=self.problem, function_name="get_instance_problem_state")  
        # 加载获取实例问题状态的函数
        self.get_solution_problem_state = load_function("problem_state.py", problem=self.problem, function_name="get_solution_problem_state") 
        # 加载获取解决方案问题状态的函数
        self.get_observation_problem_state = load_function("problem_state.py", problem=self.problem, function_name="get_observation_problem_state")  
        # 加载获取观察问题状态的函数

    def run(self, env:BaseEnv) -> bool:  # 定义运行方法，接收环境实例并返回是否成功得到有效完整解
        max_steps = int(env.construction_steps * self.iterations_scale_factor) 
        # 计算最大步骤数，为环境构建步骤数乘以缩放因子
        selection_round = 0  # 初始化选择轮次为0
        hidden_heuristics = []  # 初始化隐藏的启发式列表（暂未使用）
        heuristic_traject = []  # 初始化启发式选择轨迹列表，用于记录每轮选择和结果

        # 加载背景信息
        prompt_dict = self.llm_client.load_background(self.problem, background_file="background_without_code.txt")  # 加载问题的背景信息到提示字典

        # 生成全局启发式相关的实例问题状态
        instance_data = env.instance_data  # 获取环境中的实例数据
        instance_problem_state = self.get_instance_problem_state(instance_data)  # 调用函数获取实例问题状态
        prompt_dict["instance_problem_state"] = filter_dict_to_str([instance_data, instance_problem_state], self.problem_state_content_threshold)  
        # 将实例数据和状态过滤后存入提示字典，限制长度

        next_solution_problem_state = self.get_solution_problem_state(instance_data, env.current_solution)  
        # 获取初始解决方案的问题状态
        while selection_round * self.steps_per_selection <= max_steps and env.continue_run: 
            # 循环执行选择，直到达到最大步骤或环境停止运行
            try:  # 捕获循环中的异常
                if env.is_complete_solution:  # 如果当前是完整解，保存结果
                    env.dump_result()
                self.llm_client.load_chat("background")  # 加载背景对话历史

                # 加载启发式池文档
                heuristic_pool_doc = ""  # 初始化启发式池文档字符串
                for heuristic in self.heuristic_pool:  # 遍历启发式池
                    if heuristic not in hidden_heuristics:  # 如果启发式未被隐藏，添加其文档
                        heuristic_pool_doc += self.heuristic_docs[heuristic] + "\n"
                prompt_dict["heuristic_pool_introduction"] = heuristic_pool_doc  # 将启发式池文档存入提示字典

                # 生成当前解决方案的问题状态
                solution_data = {"current_solution": env.current_solution, env.key_item: env.key_value}  
                # 构建解决方案数据字典
                solution_problem_state = next_solution_problem_state  # 获取当前解决方案的问题状态
                prompt_dict["solution_problem_state"] = filter_dict_to_str([solution_data, solution_problem_state], self.problem_state_content_threshold)  
                # 过滤后存入提示字典

                # 生成启发式选择轨迹字符串
                if heuristic_traject == []:  # 如果轨迹为空，设为"None"
                    heuristic_trajectory_str = "None"
                else:  # 否则，取最近5轮轨迹格式化后存入
                    heuristic_trajectory_str = "\n".join([f"-----\n" + "\n".join(f"{key}: {value}" for key, value in items.items()) for items in heuristic_traject[-5:]])
                prompt_dict["discuss_round"] = str(selection_round)  # 存入当前讨论轮次
                prompt_dict["heuristic_traject"] = heuristic_trajectory_str  # 存入轨迹字符串
                prompt_dict["selection_frequency"] = self.steps_per_selection  # 存入每次选择的执行步数
                prompt_dict["num_candidate_heuristics"] = self.num_candidate_heuristics  # 存入候选启发式数量
                prompt_dict["demo_heuristic_str"] = ",".join([f"heuristic_name_{i + 1}"for i in range(self.num_candidate_heuristics)])
                # 存入候选启发式示例格式
                
                self.llm_client.load("heuristic_selection", prompt_dict)  # 加载启发式选择的提示模板和参数
                response = self.llm_client.chat()  # 与LLM交互，获取选择结果
                self.llm_client.dump(f"step_{selection_round}")  # 保存当前步骤的交互记录

                candidate_heuristics = extract(response, key="Selected heuristic", sep=",")  
                # 从LLM响应中提取候选启发式
                matched_candidate_heuristics = []  # 初始化匹配后的候选启发式列表
                for heuristic in candidate_heuristics:  # 遍历候选启发式，查找与池中名称最匹配的
                    matched_candidate_heuristic = find_closest_match(heuristic, self.heuristic_pool)
                    if matched_candidate_heuristic:  # 如果找到匹配项，添加到列表
                        matched_candidate_heuristics.append(matched_candidate_heuristic)
                assert len(matched_candidate_heuristics) > 0  # 确保至少有一个匹配的候选启发式
                
                # 通过TTS（蒙特卡洛树搜索）选择最终启发式
                selected_heuristic_name = tts_bon(
                    env,  # 环境实例
                    matched_candidate_heuristics,  # 匹配后的候选启发式
                    self.heuristic_pool,  # 启发式池
                    self.problem,  # 问题名称
                    self.iterations_scale_factor,  # 迭代缩放因子
                    self.steps_per_selection,  # 每次选择的执行步数
                    self.rollout_budget,  # 蒙特卡洛预算
                )
                # 记录选择和观察结果
                pre_observation = self.get_observation_problem_state(solution_problem_state)  
                # 获取选择前的观察状态
                pre_observation[env.key_item] = env.key_value  # 添加关键指标到观察状态
                for _ in range(self.steps_per_selection):  # 执行选定的启发式指定步数
                    env.run_heuristic(self.heuristic_functions[selected_heuristic_name], add_record_item={"step": selection_round})
                next_solution_problem_state = self.get_solution_problem_state(instance_data, env.current_solution)  
                # 获取执行后的解决方案问题状态
                next_observation = self.get_observation_problem_state(next_solution_problem_state)  
                # 获取执行后的观察状态
                next_observation[env.key_item] = env.key_value  # 添加关键指标到观察状态
                heuristic_dict = {  # 构建当前轮次的启发式选择记录字典
                    "Selection Index": selection_round,  # 选择轮次
                    "Heuristic": selected_heuristic_name,  # 选定的启发式名称
                }
                for key in pre_observation.keys():  # 记录每个观察指标的变化
                    heuristic_dict["Delta of " + key] = f"From {pre_observation[key]} to {next_observation[key]}"
                heuristic_traject.append(heuristic_dict)  # 将记录添加到轨迹列表
                selection_round += 1  # 选择轮次加1
            except Exception as e:  # 捕获异常并打印堆栈信息
                trace_string = traceback.format_exc()
                print(trace_string)
        return env.is_complete_solution and env.is_valid_solution  # 返回是否得到完整且有效的解决方案