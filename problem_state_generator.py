# 导入所需的模块：os用于文件路径操作，yaml用于处理yaml格式数据，importlib用于动态导入模块，traceback用于捕获和格式化异常信息
import os
import yaml
import importlib
import traceback
# 从自定义工具模块导入工具函数：extract用于从文本中提取特定内容，load_function用于加载函数，
# parse_text_to_dict用于将文本解析为字典，search_file用于搜索文件路径
from src.util.util import extract, load_function, parse_text_to_dict, search_file
# 从自定义的LLM客户端基类模块导入BaseLLMClient，用于与大语言模型交互
from src.util.llm_client.base_llm_client import BaseLLMClient


# 定义ProblemStateGenerator类，用于生成问题状态相关代码和描述
class ProblemStateGenerator:
    # 类的初始化方法，接收llm_client（LLM客户端实例）和problem（问题名称）作为参数
    def __init__(
        self,
        llm_client: BaseLLMClient,
        problem: str
    ) -> None:
        # 将传入的llm_client赋值给实例变量，用于后续与LLM交互
        self.llm_client = llm_client
        # 将传入的problem赋值给实例变量，标识当前处理的问题
        self.problem = problem
        # 从llm_client中获取输出目录，用于保存生成的文件
        self.output_dir = self.llm_client.output_dir
        # 创建输出目录（如果不存在），exist_ok=True确保目录已存在时不会报错
        os.makedirs(self.output_dir, exist_ok=True)

    # 生成问题状态的核心方法，返回生成的problem_state.py文件路径，smoke_test控制是否进行冒烟测试，max_try_times为最大尝试次数
    def generate_problem_state(self, smoke_test: bool=False, max_try_times: int=5) -> str:
        # 调用llm_client的load_background方法加载无代码的背景信息（问题描述等），返回包含背景信息的字典
        prompt_dict = self.llm_client.load_background(self.problem, "background_without_code")
        # 搜索当前问题的problem_state_description.txt文件路径，该文件描述问题状态的格式要求
        problem_state_description_file = search_file("problem_state_description.txt", problem=self.problem)
        # 断言文件存在，若不存在则抛出异常提示
        assert problem_state_description_file is not None, f"Problem state description file {problem_state_description_file} does not exist"
        # 读取文件内容并解析为字典，获取问题状态描述的具体内容
        problem_state_description = parse_text_to_dict(open(problem_state_description_file).read())

        # 从解析后的字典中提取实例数据介绍，存入prompt_dict供后续生成提示使用
        prompt_dict["instance_data_introduction"] = problem_state_description["instance_data"]
        # 处理key_item，提取关键项名称（去除特殊字符和空格）
        prompt_dict["key_item"] = problem_state_description["key_item"].split(":")[0].split("(")[0].replace("-", "").replace(" ", "").replace("\"", "").replace("\"", "")
        # 提取key_item的描述信息，存入prompt_dict
        prompt_dict["key_item_description"] = problem_state_description["key_item"].split(":")[-1]

        # 加载生成实例问题状态的提示，调用LLM生成响应
        self.llm_client.load("generate_instance_problem_state", prompt_dict)
        response = self.llm_client.chat()
        # 从响应中提取实例问题状态列表，处理后存入prompt_dict
        instance_problem_states = extract(response, "instance_problem_state", "\n")
        instance_problem_states = ",".join([instance_problem_state.split(";")[0] for instance_problem_state in instance_problem_states])
        prompt_dict["instance_problem_states"] = instance_problem_states

        # 加载生成实例问题状态代码的提示，调用LLM生成代码
        self.llm_client.load("implement_instance_problem_state_code", prompt_dict)
        response = self.llm_client.chat()
        # 提取生成的实例问题状态代码
        instance_problem_state_code = extract(response, "python_code")
        # 将当前对话内容 dump 到输出目录，文件名为instance_problem_state
        self.llm_client.dump(f"instance_problem_state")

        # 加载生成解决方案问题状态的提示，调用LLM生成响应
        self.llm_client.load("generate_solution_problem_state", prompt_dict)
        response = self.llm_client.chat()
        # 从响应中提取解决方案问题状态列表，处理后存入prompt_dict
        solution_problem_states = extract(response, "solution_problem_state", "\n")
        solution_problem_states = ",".join([solution_problem_state.split(";")[0] for solution_problem_state in solution_problem_states])
        prompt_dict["solution_problem_states"] = solution_problem_states

        # 加载生成解决方案问题状态代码的提示，调用LLM生成代码
        self.llm_client.load("implement_solution_problem_state_code", prompt_dict)
        response = self.llm_client.chat()
        # 提取生成的解决方案问题状态代码，并添加必要的导入语句
        solution_problem_state_code = extract(response, "python_code")
        solution_problem_state_code = f"from src.problems.{self.problem}.components import Solution\n" + solution_problem_state_code
        # 将当前对话内容 dump 到输出目录，文件名为solution_problem_state
        self.llm_client.dump(f"solution_problem_state")

        # 加载生成观察问题状态的提示，调用LLM生成响应
        self.llm_client.load("generate_observation_problem_state", prompt_dict)
        response = self.llm_client.chat()
        # 从响应中提取观察问题状态列表，处理后存入prompt_dict
        observation_problem_states = extract(response, "observation_problem_state", "\n")
        observation_problem_states = ",".join([observation_problem_state.split(";")[0] for observation_problem_state in observation_problem_states])
        prompt_dict["observation_problem_states"] = observation_problem_states

        # 加载生成观察问题状态代码的提示，调用LLM生成代码
        self.llm_client.load("implement_observation_problem_state_code", prompt_dict)
        response = self.llm_client.chat()
        # 提取生成的观察问题状态代码
        observation_problem_state_code = extract(response, "python_code")
        # 将当前对话内容 dump 到输出目录，文件名为observation_problem_state
        self.llm_client.dump(f"observation_problem_state")

        # 如果启用冒烟测试，进行代码验证和修订
        if smoke_test:
            # 最多尝试max_try_times次
            for _ in range(max_try_times):
                # 调用smoke_test方法测试三段代码，获取错误信息
                instance_error_message, solution_error_message, observation_error_message = self.smoke_test(instance_problem_state_code, solution_problem_state_code, observation_problem_state_code)
                # 当存在错误信息时，循环修正代码
                while instance_error_message or solution_error_message or observation_error_message:
                    # 将当前对话 dump 为problem_state_revision
                    self.llm_client.dump(f"problem_state_revision")
                    # 如果实例问题状态代码有错误，加载错误信息并让LLM修正代码
                    if instance_error_message:
                        self.llm_client.load(instance_error_message)
                        response = self.llm_client.chat()
                        instance_problem_state_code = extract(response, "python_code")
                    # 如果解决方案问题状态代码有错误，加载错误信息并让LLM修正代码
                    if solution_error_message:
                        self.llm_client.load(solution_error_message)
                        response = self.llm_client.chat()
                        solution_problem_state_code = extract(response, "python_code")
                    # 如果观察问题状态代码有错误，加载错误信息并让LLM修正代码
                    if observation_error_message:
                        self.llm_client.load(observation_error_message)
                        response = self.llm_client.chat()
                        observation_problem_state_code = extract(response, "python_code")
                    # 重新测试修正后的代码，获取新的错误信息
                    instance_error_message, solution_error_message, observation_error_message = self.smoke_test(instance_problem_state_code, solution_problem_state_code, observation_problem_state_code)
                # 如果仍有错误，dump为problem_state_abandoned并返回None
                if instance_error_message or solution_error_message or observation_error_message:
                    self.llm_client.dump(f"problem_state_abandoned")
                    return None
        # 保存问题状态代码到文件
        problem_state_code_file = os.path.join(self.output_dir, "problem_state.py")
        # 构建代码头部，包含生成说明和必要的导入语句
        node = f"# This file is generated by generate_problem_state.py.\n\nfrom src.problems.{self.problem}.components import Solution\n"
        # 组合三段代码为完整的problem_state.py内容
        problem_state_code = "\n\n".join([node, instance_problem_state_code, solution_problem_state_code, observation_problem_state_code])
        # 写入文件
        with open(problem_state_code_file, "w") as fp:
            fp.write(problem_state_code)
        # 打印保存路径
        print(f"Save problem state in {problem_state_code_file}")

        # 生成并保存问题状态描述
        instance_problem_state_description = self.get_problem_state_description(instance_problem_state_code)
        solution_problem_state_description = self.get_problem_state_description(solution_problem_state_code)
        # 更新问题状态描述字典
        problem_state_description["instance_problem_state"] = instance_problem_state_description
        problem_state_description["solution_problem_state"] = solution_problem_state_description
        # 处理描述内容，整理格式
        problem_state_descriptions = [line.lstrip() for line in "\n".join(problem_state_description.values()).split("\n")]
        node = "problem_state (dict): The dictionary contains the problem state with:\n    "
        problem_state_description_str = node + "\n    ".join(problem_state_descriptions)
        # 保存描述到problem_state.txt文件
        problem_state_description_file = os.path.join(self.output_dir, "problem_state.txt")
        with open(problem_state_description_file, "w") as fp:
            fp.write(problem_state_description_str)
        # 返回生成的problem_state.py文件路径
        return problem_state_code_file

    # 从问题状态代码中提取描述信息的方法
    def get_problem_state_description(self, problem_state_code: str) -> None:
        # 从代码的文档字符串中提取问题状态描述
        description = problem_state_code.split("\"\"\"")[1].split("problem state with:\n")[-1].strip()
        return description


    # 冒烟测试方法，测试三段代码是否能正常运行，返回错误信息
    def smoke_test(self, instance_problem_state_code: str, solution_problem_state_code: str, observation_problem_state_code: str) -> str:
        # 搜索冒烟测试数据所在目录
        smoke_data_dir = search_file("smoke_data", problem=self.problem)
        # 读取之前的操作记录（如果存在）
        previous_operations = []
        if os.path.exists(os.path.join(smoke_data_dir, "previous_operations.txt")):
            previous_operations = open(os.path.join(smoke_data_dir, "previous_operations.txt")).readlines()
        # 获取冒烟测试数据文件（取目录中除previous_operations.txt外的第一个文件）
        smoke_data = [file for file in os.listdir(smoke_data_dir) if file != "previous_operations.txt"][0]
        smoke_data = os.path.join(smoke_data_dir, smoke_data)

        # 准备环境：动态导入当前问题的Env类
        module = importlib.import_module(f"src.problems.{self.problem}.env")
        globals()["Env"] = getattr(module, "Env")
        # 导入当前问题的组件类（如Solution、Operator等）
        if os.path.exists(os.path.join("src", "problems", self.problem, "components.py")):
            module = importlib.import_module(f"src.problems.{self.problem}.components")
        else:
            module = importlib.import_module(f"src.problems.base.mdp_components")
        # 将组件类导入到全局变量，方便后续使用
        names_to_import = (name for name in dir(module) if not name.startswith('_'))
        for name in names_to_import:
            globals()[name] = getattr(module, name)
        # 初始化环境并重置
        env = Env(data_name=smoke_data)
        env.reset()
        # 执行之前的操作，恢复到特定状态
        for previous_operation in previous_operations:
            env.run_operator(eval(previous_operation.strip()))
        try:
            # 加载并运行实例问题状态函数，验证是否正常返回结果
            get_instance_problem_state = load_function(instance_problem_state_code, function_name="get_instance_problem_state")
            instance_problem_state = get_instance_problem_state(env.instance_data)
            assert instance_problem_state is not None
        except Exception as e:
            # 捕获异常，生成错误信息返回
            error_message = traceback.format_exc()
            return f"We got error when run get_instance_problem_state:\n{error_message}. Please fix up the get_instance_problem_state function in same format.", None, None
        try:
            # 加载并运行解决方案问题状态函数，验证是否正常返回结果
            get_solution_problem_state = load_function(solution_problem_state_code, function_name="get_solution_problem_state")
            solution_problem_state = get_solution_problem_state(env.instance_data, env.current_solution)
            assert solution_problem_state is not None
        except Exception as e:
            # 捕获异常，生成错误信息返回
            error_message = traceback.format_exc()
            return None, f"We got error when run get_solution_problem_state:\n{error_message}. Please fix up the get_solution_problem_state function in same format.", None
        try:
            # 加载并运行观察问题状态函数，验证是否正常返回结果
            get_observation_problem_state = load_function(observation_problem_state_code, function_name="get_observation_problem_state")
            observation_problem_state = get_observation_problem_state(solution_problem_state)
            assert observation_problem_state is not None
        except Exception as e:
            # 捕获异常，生成错误信息返回
            error_message = traceback.format_exc()
            return None, None, f"We got error when run get_observation_problem_state:\n{error_message}. Please fix up the get_observation_problem_state function in same format."
        # 若所有测试通过，返回None（无错误）
        return None, None, None