import os  # 导入os模块，用于文件和目录操作
import json  # 导入json模块，用于JSON数据的处理
import importlib  # 导入importlib模块，用于动态导入模块
import traceback  # 导入traceback模块，用于捕获和输出异常信息
from copy import deepcopy  # 从copy模块导入deepcopy，用于深拷贝对象
from src.problems.base.components import BaseOperator  # 从基础组件模块导入BaseOperator类，作为操作的基类
from src.util.util import (  # 从工具模块导入多个工具函数，用于提取信息、加载函数、处理字典等
    extract, extract_function_with_short_docstring, filter_dict_to_str, 
    find_key_value, load_function, parse_paper_to_dict, replace_strings_in_dict, 
    sanitize_function_name, load_framework_description, search_file
)
from src.util.llm_client.base_llm_client import BaseLLMClient  # 从LLM客户端模块导入BaseLLMClient类，作为LLM客户端的基类


class HeuristicGenerator:  # 定义启发式生成器类，用于生成启发式算法
    def __init__(  # 初始化方法，设置LLM客户端、问题名称和输出目录
        self,
        llm_client: BaseLLMClient,  # LLM客户端实例，用于与大语言模型交互
        problem: str  # 问题名称，指定要处理的组合优化问题
    ) -> None:
        self.llm_client = llm_client  # 保存LLM客户端实例
        self.problem = problem  # 保存问题名称
        self.output_dir = self.llm_client.output_dir  # 从LLM客户端获取输出目录
        os.makedirs(self.output_dir, exist_ok=True)  # 创建输出目录（若不存在）

    def generate_from_llm(self, reference_data: str=None, smoke_test: bool=False) -> list[str]:  # 从LLM直接生成启发式算法
        heuristic_files = []  # 存储生成的启发式算法文件路径

        # 加载背景信息（问题描述、解决方案框架等）
        prompt_dict = self.llm_client.load_background(self.problem, "background_with_code", reference_data)

        # 生成可用的启发式算法描述
        self.llm_client.load("generate_from_llm", prompt_dict)  # 加载生成启发式的提示模板
        response = self.llm_client.chat()  # 与LLM交互，获取响应
        heuristics = extract(response, "heuristic", sep="\n")  # 从响应中提取启发式算法列表
        self.llm_client.dump("heuristic_from_llm")  # 保存交互记录

        for heuristic in heuristics:  # 遍历每个启发式算法描述，生成对应的代码文件
            # 加载之前的对话记录，继续生成单个启发式的详细信息
            self.llm_client.load_chat("heuristic_from_llm")
            heuristic_name = heuristic.split(":")[0]  # 提取启发式名称
            description = heuristic[len(heuristic_name) + 1: ]  # 提取启发式描述
            env_summarize = prompt_dict["env_summarize"]  # 获取环境数据摘要
            # 调用generate方法生成启发式代码文件，并添加到列表中
            heuristic_files.append(self.generate(heuristic_name, description, env_summarize, smoke_test))

        return heuristic_files  # 返回生成的启发式文件路径列表

    def generate_from_paper(self, paper_path: str,  reference_data: str=None, smoke_test: bool=False) -> str:  # 从学术论文生成启发式算法
        heuristic_file = None  # 存储生成的启发式文件路径
        # 加载背景信息
        prompt_dict = self.llm_client.load_background(self.problem, "background_with_code", reference_data)

        # 加载论文全文内容
        if os.path.isdir(paper_path):  # 若输入是目录，拼接所有tex文件内容
            paper_content = ""
            for file in os.listdir(paper_path):
                if file.split(".")[-1] == "tex":
                    paper_content += open(os.path.join(paper_path, file)).read()
        elif os.path.isfile(paper_path):  # 若输入是文件，直接读取内容
            paper_content = open(paper_path).read()

        # 将论文按章节解析为字典
        section_dict = parse_paper_to_dict(paper_content)
        # 提取论文标题（优先从字典获取，否则用文件名）
        if "Title" in section_dict:
            title = section_dict["Title"]
        else:
            title = os.path.splitext(os.path.basename(paper_path))[0]
        # 提取论文摘要（优先从字典获取，否则用引言或默认文本）
        if "Abstract" in section_dict:
            abstract = section_dict["Abstract"]
        elif "Introduction" in section_dict:
            abstract = section_dict["Introduction"]
        else:
            abstract = "No abstract found."

        # 读取摘要，判断论文是否与当前问题相关
        prompt_dict["title"] = title
        prompt_dict["abstract"] = abstract
        self.llm_client.load("reading_paper_abstract", prompt_dict)  # 加载判断相关性的提示模板
        response = self.llm_client.chat()  # 与LLM交互
        related_to_problem = extract(response, "related_to_problem")  # 提取相关性判断结果
        self.llm_client.dump("read_paper")  # 保存交互记录

        if "yes" in related_to_problem:  # 若论文与问题相关，继续处理
            last_interested_section = "None"  # 记录上一个感兴趣的章节
            last_interested_content = "None"  # 记录上一个感兴趣章节的内容
            remaining_section_dict = deepcopy(section_dict)  # 深拷贝章节字典，避免修改原数据
            remaining_section_dict = replace_strings_in_dict(remaining_section_dict)  # 替换字典中的字符串（可能用于简化显示）
            dict_str = json.dumps(remaining_section_dict, indent=4)  # 将字典转为JSON字符串
            # 循环读取章节，直到生成启发式或放弃
            while True:
                prompt_dict["last_interested_section"] = last_interested_section
                prompt_dict["last_interested_content"] = last_interested_content
                prompt_dict["remaining_section_dict"] = dict_str
                self.llm_client.load("reading_paper_section", prompt_dict)  # 加载读取章节的提示模板
                response = self.llm_client.chat()  # 与LLM交互，获取感兴趣的章节
                interested_section = extract(response, "interested_section")  # 提取感兴趣的章节
                if interested_section is None:  # 若没有感兴趣的章节，放弃
                    self.llm_client.dump("abandoned")
                    return None
                # 查找感兴趣章节的内容
                interested_content = find_key_value(section_dict, interested_section)
                if interested_content is None:  # 若章节内容不存在，生成启发式代码
                    self.llm_client.dump(f"generate_from_paper")
                    heuristic_name = interested_section
                    env_summarize = prompt_dict["env_summarize"]
                    # 调用generate方法生成代码文件
                    heuristic_file = self.generate(heuristic_name, f"Generate from paper {title}. Please add the notes in code to show the source paper.", env_summarize, smoke_test)
                    return heuristic_file
                # 更新上一个感兴趣的章节及内容，继续循环
                last_interested_section = interested_section
                last_interested_content = interested_content
        else:  # 若论文与问题无关，放弃
            self.llm_client.dump(f"abandoned")
            return None

    def generate_from_reference(self, related_problems: list[str], reference_data: str=None, smoke_test: bool=False) -> list[str]:  # 从相关问题迁移启发式算法
        heuristic_files = []  # 存储生成的启发式文件路径

        # 加载背景信息
        prompt_dict = self.llm_client.load_background(self.problem, "background_with_code", reference_data)

        # 查找相似问题的描述
        description_dict = {}
        for problem in related_problems:
            # 搜索相关问题的描述文件
            related_problem_description_file = search_file("problem_description.txt", problem=problem)
            if related_problem_description_file:
                description_dict[problem] = open(related_problem_description_file).read()  # 读取描述内容

        # 拼接所有相关问题的描述，用于提示LLM
        studied_problems = "\n\n".join([
            f"problem name: {problem}\ndescription: {description_dict[problem]}"
            for problem in related_problems
        ])
        prompt_dict["studied_problems"] = studied_problems
        self.llm_client.load("reference_problem", prompt_dict)  # 加载查找参考问题的提示模板
        response = self.llm_client.chat()  # 与LLM交互
        related_problems = extract(response, "referenced_problem", ";")  # 提取相关问题列表
        self.llm_client.dump("reference_problem")  # 保存交互记录

        for referenced_problem in related_problems:  # 遍历每个相关问题
            if referenced_problem not in description_dict:  # 若问题描述不存在，跳过
                continue
            self.llm_client.load_chat("reference_problem")  # 加载之前的对话记录

            # 查找参考问题与当前问题的相似性
            # 读取参考问题的组件代码
            component_code = open(os.path.join("src", "problems", referenced_problem, "components.py")).read()
            # 提取参考问题的解决方案和操作类的框架描述
            reference_solution_class, reference_operation_class = load_framework_description(component_code)
            description = description_dict[referenced_problem]  # 参考问题的描述
            prompt_dict["referenced_problem"] = referenced_problem
            prompt_dict["referenced_problem_description"] = description
            prompt_dict["referenced_problem_solution_class"] = reference_solution_class
            prompt_dict["referenced_problem_operation_class"] = reference_operation_class
            self.llm_client.load("mapping_component_in_problem", prompt_dict)  # 加载映射组件的提示模板
            response = self.llm_client.chat()  # 与LLM交互
            similarities = extract(response, "similarities", "\n")  # 提取组件相似性列表
            prompt_dict["similarities_in_problem"] = "\n".join(similarities)  # 保存相似性信息

            # 检查参考问题的启发式算法
            referenced_heuristic_dir = os.path.join("src", "problems", referenced_problem, "heuristics", "basic_heuristics")
            # 获取参考问题的所有启发式名称
            referenced_heuristic_names = [heuristic_file.split(".")[0] for heuristic_file in os.listdir(referenced_heuristic_dir)]
            referenced_heuristic_docs = []  # 存储参考启发式的文档字符串
            for heuristic_name in referenced_heuristic_names:
                # 提取启发式的简短文档字符串
                referenced_heuristic_doc = extract_function_with_short_docstring(open(os.path.join(referenced_heuristic_dir, heuristic_name + ".py")).read(), heuristic_name).split(":")[-1]
                referenced_heuristic_docs.append(f"{heuristic_name}:{referenced_heuristic_doc}")
            referenced_heuristic_docs = "\n".join(referenced_heuristic_docs)  # 拼接文档字符串
            prompt_dict["candidate_heuristic_pool"] = referenced_heuristic_docs
            self.llm_client.load("reference_heuristic", prompt_dict)  # 加载参考启发式的提示模板
            response = self.llm_client.chat()  # 与LLM交互
            reference_heuristics = extract(response, "referenced_heuristics", "\n")  # 提取相关启发式列表
            self.llm_client.dump(f"reference_heuristics_in_{referenced_problem}")  # 保存交互记录

            # 查找参考启发式与当前问题的相似性，并迁移生成新启发式
            for reference_heuristic_item in reference_heuristics:
                self.llm_client.load_chat(f"reference_heuristics_in_{referenced_problem}")  # 加载之前的对话记录
                reference_heuristic = reference_heuristic_item.split(";")[0]  # 提取参考启发式名称
                # 读取参考启发式的代码
                reference_heuristic_file = os.path.join("src", "problems", referenced_problem, "heuristics", "basic_heuristics", reference_heuristic + ".py")
                reference_heuristic_code = open(reference_heuristic_file).read()
                prompt_dict["referenced_heuristic"] = reference_heuristic
                prompt_dict["referenced_heuristic_code"] = reference_heuristic_code
                # 查找参考问题的状态描述文件
                referenced_problem_state_introduction = search_file("problem_state.txt", problem=referenced_problem)
                if referenced_problem_state_introduction:
                    # 读取参考问题的状态描述
                    prompt_dict["referenced_problem_state_introduction"] = open(os.path.join("src", "problems", referenced_problem, "prompt", "problem_state.txt")).read()
                self.llm_client.load("mapping_component_in_heuristic", prompt_dict)  # 加载映射启发式组件的提示模板
                response = self.llm_client.chat()  # 与LLM交互
                similarities_in_heuristics = extract(response, "similarities", "\n")  # 提取启发式层面的相似性
                if similarities_in_heuristics:
                    similarities += similarities_in_heuristics  # 合并相似性列表

                # 更新启发式描述，准备生成新代码
                self.llm_client.load("transfer_heuristic", prompt_dict)  # 加载迁移启发式的提示模板
                response = self.llm_client.chat()  # 与LLM交互
                heuristic_name, description = extract(response, "heuristic", ";")  # 提取新启发式名称和描述
                # 补充描述信息，说明迁移来源和相似性
                description += f"We hope to transfer {reference_heuristic} in {referenced_problem} into new {heuristic_name} in {self.problem}.\n" \
                    + "Following are some similarities(source_component;target_component;introduction):\n" \
                    + "\n".join(similarities)
                env_summarize = prompt_dict["env_summarize"]  # 环境数据摘要

                # 生成新启发式的代码文件
                heuristic_files.append(self.generate(heuristic_name, description, env_summarize, smoke_test))
        return heuristic_files  # 返回生成的启发式文件路径列表

    def generate(self, heuristic_name: str, description: str, env_summarize: str="All data are possible", smoke_test: bool=False, more_prompt_dict=None, reminder=True) -> str:  # 生成单个启发式算法的代码文件
        # 加载特殊提示信息（用于避免常见错误）
        special_remind_file = os.path.join("src", "problems", self.problem, "prompt", "special_remind.txt")
        special_remind = "None"
        if os.path.exists(special_remind_file):
            special_remind = open(special_remind_file).read()

        # 生成函数名称（标准化处理，避免命名冲突）
        function_name = sanitize_function_name(heuristic_name, description)
        # 构建提示字典，包含生成代码所需的信息
        prompt_dict = {"problem": self.problem, "heuristic_name": heuristic_name, "description": description, "function_name": function_name, "special_remind": special_remind, "env_summarize": env_summarize}
        if more_prompt_dict:  # 合并额外的提示信息
            prompt_dict.update(more_prompt_dict)

        # 确定组件文件路径（优先使用当前问题的组件，否则用基础组件）
        if os.path.exists(os.path.join("src", "problems", self.problem, "components.py")):
            prompt_dict["components_file"] = f"src.problems.{self.problem}.components"
        else:
            prompt_dict["components_file"] = f"src.problems.base.mdp_components"
        # 根据是否需要提醒，加载对应的代码生成提示模板
        if reminder:
            self.llm_client.load("implement_code_with_reminder", prompt_dict)
        else:
            self.llm_client.load("implement_code_without_reminder", prompt_dict)
        response = self.llm_client.chat()  # 与LLM交互，获取生成的代码
        code = extract(response, "python_code")  # 提取代码内容

        # 若需要冒烟测试，验证并修正代码
        if smoke_test:
            code = self.smoke_test(code, function_name)
            if not code:  # 若测试失败，放弃生成
                self.llm_client.dump(f"{function_name}_abandoned")
                return None

        self.llm_client.dump(f"{function_name}")  # 保存交互记录

        # 保存生成的代码到文件
        output_heuristic_file = os.path.join(self.output_dir, function_name + ".py")
        print(f"Save {function_name} code to {output_heuristic_file}")
        with open(output_heuristic_file, "w") as fp:
            fp.write(code)
        return output_heuristic_file  # 返回生成的文件路径

    def smoke_test(self, heuristic_code: str, function_name: str, max_try_times: int=5) -> str:  # 对生成的启发式代码进行冒烟测试
        prompt_dict = {}
        # 确定组件文件路径
        if os.path.exists(os.path.join("src", "problems", self.problem, "components.py")):
            prompt_dict["components_file"] = f"src.problems.{self.problem}.components"
        else:
            prompt_dict["components_file"] = f"src.problems.base.mdp_components"
        # 加载冒烟测试数据
        smoke_data_dir = search_file("smoke_data", problem=self.problem)
        previous_operations = open(os.path.join(smoke_data_dir, "previous_operations.txt")).readlines()  # 读取之前的操作记录
        # 获取测试数据文件（排除操作记录文件）
        smoke_data = [file for file in os.listdir(smoke_data_dir) if file != "previous_operations.txt"][0]
        smoke_data = os.path.join(smoke_data_dir, smoke_data)
        prompt_dict["function_name"] = function_name
        prompt_dict["previous_operations"] = "".join(previous_operations)

        # 准备测试环境
        module = importlib.import_module(f"src.problems.{self.problem}.env")  # 导入环境模块
        globals()["Env"] = getattr(module, "Env")  # 将环境类加入全局变量
        # 导入组件模块（优先当前问题，否则用基础组件）
        if os.path.exists(os.path.join("src", "problems", self.problem, "components.py")):
            module = importlib.import_module(f"src.problems.{self.problem}.components")
        else:
            module = importlib.import_module(f"src.problems.base.mdp_components")
        # 将组件类加入全局变量
        names_to_import = (name for name in dir(module) if not name.startswith('_'))
        for name in names_to_import:
            globals()[name] = getattr(module, name)
        env = Env(data_name=smoke_data)  # 初始化环境
        # 最多尝试max_try_times次修正代码
        for _ in range(max_try_times):
            env.reset()  # 重置环境
            # 提取实例问题状态并过滤（简化显示）
            prompt_dict["smoke_instance_problem_state"] = filter_dict_to_str(env.get_instance_problem_state(env.instance_data))
            # 执行之前的操作，恢复到测试状态
            for previous_operation in previous_operations:
                env.run_operator(eval(previous_operation.strip()))
            prompt_dict["smoke_solution"] = env.current_solution  # 当前解决方案
            # 提取解决方案状态并过滤
            prompt_dict["smoke_solution_problem_state"] = filter_dict_to_str(env.get_solution_problem_state(env.instance_data, env.current_solution))
            try:
                # 加载启发式函数并运行一次
                heuristic = load_function(heuristic_code, function_name=function_name)
                operator = env.run_heuristic(heuristic)
            except Exception as e:  # 捕获运行时异常
                operator = traceback.format_exc()  # 保存异常信息
            # 若操作有效（为空或BaseOperator实例），进行结果对比
            if operator is None or isinstance(operator, BaseOperator):
                # 获取预期结果
                self.llm_client.load("smoke_test_expected_result.txt", prompt_dict)
                response = self.llm_client.chat()
                expected_result = extract(response, "expected_result")

                # 准备实际结果信息
                prompt_dict["output_result"] = str(operator)
                prompt_dict["updated_smoke_solution"] = env.current_solution
                prompt_dict["updated_smoke_solution_problem_state"] = filter_dict_to_str(env.get_solution_problem_state(env.instance_data, env.current_solution))

                # 对比实际结果与预期结果
                prompt_dict["expected_result"] = expected_result
                self.llm_client.load("smoke_test_compare.txt", prompt_dict)
                response = self.llm_client.chat()
                response = extract(response, "python_code")
                # 根据对比结果处理
                if response is None:  # 无法修正，放弃
                    self.llm_client.load("We can not implement and give up.")
                    return None
                elif "correct" in response:  # 结果正确，返回代码
                    self.llm_client.load(f"To ensure the stable of heuristics, we adjust the code to:\n{heuristic_code}")
                    return heuristic_code
                else:  # 结果不正确，更新代码重试
                    heuristic_code = response
            else:  # 代码运行崩溃，提示LLM修正
                prompt_dict["error_message"] = operator
                self.llm_client.load("smoke_test_crashed.txt", prompt_dict)
                response = self.llm_client.chat()
                heuristic_code = extract(response, "python_code")
                if heuristic_code is None:  # 无法修正，放弃
                    self.llm_client.load("We can not implement and give up.")
                    return None
        # 超过最大尝试次数，放弃
        self.llm_client.load("We can not implement and give up.")
        return None