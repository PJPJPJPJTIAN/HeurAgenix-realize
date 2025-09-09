# 导入必要的模块：os用于文件路径操作，json用于JSON数据处理，re用于正则表达式匹配，
# base64用于图片编码，importlib用于动态导入模块，time.sleep用于延迟操作，以及自定义工具函数
import os
import json
import re
import base64
import importlib
from time import sleep
from src.util.util import compress_numbers, extract, load_framework_description, search_file

# 定义BaseLLMClient基类，作为所有LLM客户端的抽象基类，提供通用的初始化、聊天、加载/保存对话等方法
class BaseLLMClient:
    # 初始化方法：接收配置字典、提示词目录和输出目录，初始化实例变量并重置对话状态
    def __init__(
            self,
            config: dict,
            prompt_dir: str=None,
            output_dir: str=None,
        ):
        self.prompt_dir = prompt_dir  # 提示词文件所在目录
        self.output_dir = output_dir  # 对话输出文件保存目录
        self.config = config  # LLM配置字典（包含模型类型、API参数等）
        self.reset(output_dir)  # 重置对话状态，同时确保输出目录存在

    # 重置对话状态：清空消息列表，若指定新输出目录则更新并创建目录
    def reset(self, output_dir:str=None) -> None:
        self.messages = []  # 存储对话历史的列表，每个元素为包含"role"和"content"的字典
        if output_dir is not None:
            self.output_dir = output_dir
            os.makedirs(output_dir, exist_ok=True)  # 确保输出目录存在，不存在则创建

    # 单次聊天方法（抽象方法，需子类实现）：用于与LLM进行一次交互并返回响应内容
    def chat_once(self) -> str:
        pass  # 子类需重写此方法实现具体的LLM调用逻辑

    # 聊天方法：循环调用chat_once，支持失败重试，直到获取响应或达到最大尝试次数
    def chat(self) -> str:
        # 从配置中获取最大尝试次数和重试间隔时间，默认为50次和10秒
        self.max_attempts = self.config.get("max_attempts", 50)
        self.sleep_time = self.config.get("sleep_time", 10)
        for index in range(self.max_attempts):
            try:
                response_content = self.chat_once()  # 调用单次聊天方法
                # 将LLM的响应添加到对话历史中，角色为"assistant"，内容为文本类型
                self.messages.append({"role": "assistant", "content": [{"type": "text", "text": response_content}]})
                return response_content  # 返回成功获取的响应内容
            except Exception as e:
                print(f"Try to chat {index + 1} time: {e}")  # 打印重试信息和错误
                sleep(self.sleep_time)  # 等待指定时间后重试
        # 若超过最大尝试次数，添加错误信息到对话历史并保存
        self.messages.append({"role": "assistant", "content": "Exceeded the maximum number of attempts"})
        self.dump("error")

    # 加载聊天历史：从JSON文件中读取对话历史并覆盖当前messages
    def load_chat(self, chat_file: str) -> None:
        # 确保文件名以.json结尾
        if chat_file.split(".")[-1] != "json":
            chat_file = chat_file + ".json"
        # 查找聊天文件：优先从提示词目录查找，再从输出目录查找
        if self.prompt_dir is not None and os.path.exists(os.path.join(self.prompt_dir, chat_file)):
            chat_file = os.path.join(self.prompt_dir, chat_file)
        elif self.prompt_dir is not None and os.path.exists(os.path.join(self.output_dir, chat_file)):
            chat_file = os.path.join(self.output_dir, chat_file)
        # 读取并加载JSON格式的对话历史
        with open(chat_file, "r") as fp:
            self.messages = json.load(fp)

    # 加载问题背景信息：构建包含问题描述、解决方案类、操作符类等的提示词字典，并验证问题是否为组合优化问题
    def load_background(self, problem: str, background_file="background_with_code", reference_data: str=None) -> dict:
        # 确定问题的提示词目录
        problem_dir = os.path.join("src", "problems", problem, "prompt")
        # 加载问题的组件代码（components.py），若不存在则使用基础模板（mdp_components.py）
        if os.path.exists(os.path.join("src", "problems", problem, "components.py")):
            component_code = open(os.path.join("src", "problems", problem, "components.py")).read()
        else:
            component_code = open(os.path.join("src", "problems", "base", "mdp_components.py")).read()
        # 从组件代码中提取解决方案类和操作符类的框架描述
        solution_class_str, operator_class_str = load_framework_description(component_code)

        # 总结参考数据环境信息，默认为"All data is possible"
        env_summarize = "All data is possible"
        if reference_data:
            # 动态导入问题的Env类，加载参考数据并生成环境总结
            module = importlib.import_module(f"src.problems.{problem}.env")
            globals()["Env"] = getattr(module, "Env")
            env = Env(reference_data)
            env_summarize = env.summarize_env()
        # 查找问题描述文件和问题状态文件
        problem_description_file = search_file("problem_description.txt", problem)
        problem_state_file = search_file("problem_state.txt", problem)
        # 断言问题描述目录存在，否则抛出异常
        assert os.path.exists(problem_dir), f"Problem description file {problem_description_file} does not exist"

        # 构建提示词字典，包含问题相关的各种信息
        prompt_dict = {
            "problem": problem,  # 问题名称
            "problem_description": open(problem_description_file, encoding="utf-8").read(),  # 问题描述内容
            # 问题状态介绍（若文件存在则读取，否则为空）
            "problem_state_introduction": open(problem_state_file, encoding="utf-8").read() if problem_state_file else "",
            "solution_class": solution_class_str,  # 解决方案类的框架描述
            "operator_class": operator_class_str,  # 操作符类的框架描述
            "env_summarize": env_summarize  # 环境数据总结
        }

        # 加载背景提示词模板并发送给LLM，验证问题是否为组合优化问题
        self.load(background_file, prompt_dict)
        response = self.chat()
        is_cop = extract(response, "is_cop", "\n")  # 提取LLM对问题类型的判断
        self.dump("background")  # 保存背景对话
        # 若判断结果为非组合优化问题，抛出异常
        if not is_cop or "no" in is_cop or "No" in is_cop or "NO" in is_cop:
            raise BaseException("Not combination optimization problem")
        return prompt_dict  # 返回构建的提示词字典

    # 加载提示词模板：读取指定的提示词文件，替换占位符，并添加到对话历史中
    def load(self, message: str, replace: dict={}) -> None:
        # 查找提示词文件：优先从提示词目录查找，支持带.txt后缀或不带的文件名
        if self.prompt_dir is not None and os.path.exists(os.path.join(self.prompt_dir, message)):
            message = open(os.path.join(self.prompt_dir, message), "r", encoding="UTF-8").read()
        elif self.prompt_dir is not None and os.path.exists(os.path.join(self.prompt_dir, message)):
            message = open(os.path.join(self.prompt_dir, message), "r", encoding="UTF-8").read()
        elif self.prompt_dir is not None and os.path.exists(os.path.join(self.prompt_dir, message + ".txt")):
            message = open(os.path.join(self.prompt_dir, message + ".txt"), "r", encoding="UTF-8").read()
        elif os.path.exists(message):
            message = open(message, "r", encoding="UTF-8").read()
        elif os.path.exists(message + ".txt"):
            message = open(message + ".txt", "r", encoding="UTF-8").read()
        # 替换提示词中的占位符（如{problem}、{description}等）
        for key, value in replace.items():
            if value is None or str(value) == "":
                value = "None"  # 空值替换为"None"
            message = message.replace("{" + key + "}", str(value))
        # 处理提示词中的图片引用（格式为[image: 图片路径]）
        image_key = r"\[image: (.*?)\]"
        texts = re.split(image_key, message)  # 分割文本和图片路径
        images = re.compile(image_key).findall(message)  # 提取所有图片路径
        current_message = []  # 构建当前消息内容列表
        for i in range(len(texts)):
            if i % 2 == 1:  # 奇数索引对应图片路径
                # 读取图片并进行base64编码
                encoded_image = base64.b64encode(open(images[int((i - 1)/ 2)], 'rb').read()).decode('ascii')
                current_message.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"},  # 图片的base64 URL
                    "image_path": images[int((i - 1)/ 2)]  # 原始图片路径
                })
            else:  # 偶数索引对应文本内容
                current_message.append({
                    "type": "text",
                    "text": compress_numbers(texts[i])  # 压缩文本中的数字格式
                })
        # 将处理后的消息添加到对话历史，角色为"user"
        self.messages.append({"role": "user", "content": current_message})

    # 保存对话历史：将当前messages保存为JSON文件和文本文件，返回最后一条消息的文本内容
    def dump(self, output_name: str=None) -> str:
        if self.output_dir != None and output_name != None:
            # 定义JSON和文本输出文件路径
            json_output_file = os.path.join(self.output_dir, f"{output_name}.json")
            text_output_file = os.path.join(self.output_dir, f"{output_name}.txt")
            print(f"Chat dumped to {text_output_file}")  # 打印保存路径
            # 保存为JSON文件（保留完整结构）
            with open(json_output_file, "w") as fp:
                json.dump(self.messages, fp, indent=4)

            # 保存为文本文件（便于人类阅读）
            with open(text_output_file, "w", encoding="UTF-8") as file:
                for message in self.messages:
                    file.write(message["role"] + "\n")  # 写入角色（user/assistant）
                    contents = ""
                    for i, content in enumerate(message["content"]):
                        if content["type"] == "image_url":
                            contents += f"[image: {content['image_path']}]"  # 图片用路径标识
                        else:
                            contents += content["text"]  # 文本直接写入
                    # 用分隔线分隔不同消息
                    file.write(contents + "\n------------------------------------------------------------------------------------\n\n")
        # 返回最后一条消息的文本内容（若存在）
        return self.messages[-1]["content"][0]["text"] if self.messages else ""