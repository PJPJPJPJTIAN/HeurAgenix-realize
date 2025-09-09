# 导入必要的模块：os用于文件操作，json用于JSON数据处理，requests用于发送HTTP请求，以及从基础类模块导入BaseLLMClient基类
import os
import json
import requests
from src.util.llm_client.base_llm_client import BaseLLMClient

# 定义APIModelClient类，继承自BaseLLMClient，用于通过API接口与大语言模型交互
class APIModelClient(BaseLLMClient):
    # 初始化方法：接收配置字典、提示词目录和输出目录，调用父类初始化方法并设置API相关参数
    def __init__(
            self,
            config: dict,
            prompt_dir: str=None,
            output_dir: str=None,
        ):
        super().__init__(config, prompt_dir, output_dir)  # 调用父类的初始化方法

        self.url = config["url"]  # 从配置中获取API请求地址
        model = config["model"]  # 从配置中获取模型名称
        stream = config.get("stream", False)  # 从配置中获取流式输出标志，默认为False
        top_p = config.get("top-p", 0.7)  # 从配置中获取top-p参数，默认为0.7
        temperature = config.get("temperature", 0.95)  # 从配置中获取温度参数，默认为0.95
        max_tokens = config.get("max_tokens", 3200)  # 从配置中获取最大token数，默认为3200
        seed = config.get("seed", None)  # 从配置中获取随机种子，默认为None
        api_key = config["api_key"]  # 从配置中获取API密钥
        self.max_attempts = config.get("max_attempts", 50)  # 从配置中获取最大尝试次数，默认为50
        self.sleep_time = config.get("sleep_time", 60)  # 从配置中获取重试间隔时间，默认为60秒
        self.headers = {  # 定义HTTP请求头，包含认证信息和内容类型
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        self.payload = {  # 定义请求体参数，包含模型配置信息
            "model": model,
            "stream": stream,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "seed": seed
        }

    # 重置对话状态：清空消息列表，若指定新输出目录则更新并创建目录（重写父类方法）
    def reset(self, output_dir:str=None) -> None:
        self.messages = []  # 清空存储对话历史的列表
        if output_dir is not None:
            self.output_dir = output_dir  # 更新输出目录
            os.makedirs(output_dir, exist_ok=True)  # 确保输出目录存在，不存在则创建

    # 单次聊天方法（重写父类抽象方法）：发送当前对话历史到API并返回模型响应
    def chat_once(self) -> str:
        self.payload["messages"] = self.messages  # 将对话历史添加到请求体中
        response = requests.request("POST", self.url, json=self.payload, headers=self.headers)  # 发送POST请求到API
        response_content = json.loads(response.text)["choices"][-1]["message"]["content"]  # 解析响应，提取最后一个选择的消息内容
        return response_content  # 返回模型响应的文本内容