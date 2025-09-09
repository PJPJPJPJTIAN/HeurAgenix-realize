# 导入json模块用于处理JSON格式的配置文件，从src.util.llm_client.base_llm_client模块导入BaseLLMClient基类（所有LLM客户端的基础类）
import json
from src.util.llm_client.base_llm_client import BaseLLMClient

# 定义get_llm_client函数，用于根据配置文件创建并返回对应的大语言模型（LLM）客户端实例，参数包括配置文件路径、提示词目录和输出目录
def get_llm_client(
        config_file: str,
        prompt_dir: str=None,
        output_dir: str=None,
        ) -> BaseLLMClient:
    # 加载配置文件内容，将JSON格式的配置数据解析为字典
    config = json.load(open(config_file))
    # 从配置字典中获取LLM类型（如"azure_apt"、"api_model"、"local_model"）
    llm_type = config["type"]
    # 根据LLM类型创建对应的客户端实例
    if llm_type == "azure_apt":
        # 若类型为azure_apt，导入AzureGPTClient类并实例化，传入配置、提示词目录和输出目录
        from src.util.llm_client.azure_gpt_client import AzureGPTClient
        llm_client = AzureGPTClient(config=config, prompt_dir=prompt_dir, output_dir=output_dir)
    elif llm_type == "api_model":
        # 若类型为api_model，导入APIModelClient类并实例化，传入配置、提示词目录和输出目录
        from src.util.llm_client.api_model_client import APIModelClient
        llm_client = APIModelClient(config=config, prompt_dir=prompt_dir, output_dir=output_dir)
    elif llm_type == "local_model":
        # 若类型为local_model，导入LocalModelClient类并实例化，传入配置、提示词目录和输出目录
        from src.util.llm_client.local_model_client import LocalModelClient
        llm_client = LocalModelClient(config=config, prompt_dir=prompt_dir, output_dir=output_dir)
    # 返回创建的LLM客户端实例（该实例继承自BaseLLMClient，统一接口便于后续调用）
    return llm_client