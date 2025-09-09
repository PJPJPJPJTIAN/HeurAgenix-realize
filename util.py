# 导入所需模块：ast用于解析Python代码的抽象语法树，os用于文件路径操作，re用于正则表达式处理，io用于字符串与文件流转换，hashlib用于生成哈希值，numpy和pandas用于数据处理，difflib用于字符串匹配
import ast
import os
import re
import io
import hashlib
import numpy as np
import pandas as pd
import difflib

# 定义extract函数，从文本中提取指定key对应的内容，sep用于分割结果为列表
def extract(message: str, key: str, sep=None) -> list[str]:
    # 定义多种可能的正则表达式格式，用于匹配包含key的内容块（如***key:内容***等格式）
    formats = [
        rf"\*\*\*.*{key}:(.*?)\*\*\*",
        rf" \*\*\*{key}:(.*?)\*\*\*",
        rf"\*\*\*\n{key}:(.*?)\*\*\*",
        rf"\*.*{key}:(.*?)\*",
    ]
    # 替换文本中的空行为单换行，便于处理
    message.replace("\n\n", "\n")
    # 遍历每种格式进行匹配
    for format in formats:
        # 使用正则表达式搜索，re.DOTALL使.匹配包括换行符在内的所有字符
        match = re.search(format, message, re.DOTALL)
        value = None
        if match:
            # 提取匹配到的内容并去除首尾空白
            value = match.group(1).strip()
            if value:
                # 如果指定了分隔符，按分隔符分割内容并返回列表
                if sep:
                    return value.split(sep)
                else:
                    # 如果内容是None或none，返回None，否则返回提取的内容
                    if value in ["None", "none"]:
                        return None
                    return value
    # 如果没有匹配到，根据sep返回空列表或None
    if sep:
        return []
    else:
        return None

# 定义find_closest_match函数，在字符串列表中找到与输入字符串最相似的匹配项
def find_closest_match(input_string, string_list):
    # 使用difflib的get_close_matches找到最接近的匹配（最多1个，匹配度阈值0.0）
    matches = difflib.get_close_matches(input_string, string_list, n=1, cutoff=0.0)
    # 返回匹配结果（如果有），否则返回None
    return matches[0] if matches else None 

# 定义parse_text_to_dict函数，将特定格式的文本解析为字典（如以"- key: value"开头的行）
def parse_text_to_dict(text):
    # 按换行分割文本为行列表
    lines = text.split("\n")
    result = {}
    current_key = None
    current_content = []
    # 遍历每一行
    for line in lines:
        # 识别以"- "开头且包含":"的行作为键的开始
        if len(line) > 0 and line[0] == "-" and ":" in line:
            # 如果已有当前键，将积累的内容存入结果字典
            if current_key:
                result[current_key.replace(" ", "")] = "\n".join(current_content).strip()
            # 提取新的键（去除开头的"- "并按":"分割）
            current_key = line[1:].split(":")[0]
            current_content = []
            # 提取键对应的初始内容（按":"分割后的部分）
            if len(line.split(":")) > 0:
                current_content = [line.split(":")[1]]
        # 如果已有当前键，将行内容添加到当前内容列表
        elif current_key:
            current_content.append(line)
    # 处理最后一个键值对
    if current_key:
        result[current_key.replace(" ", "")] = "\n".join(current_content).strip()
    return result

# 定义load_function函数，从文件或代码字符串中加载指定名称的函数并返回
def load_function(file:str, problem: str="base", function_name: str=None) -> callable:
    # 如果输入不是代码字符串（不含换行），则视为文件名处理
    if not "\n" in file:
        # 确保文件名以.py结尾
        if not file.endswith(".py"):
            file += ".py"
        # 搜索文件路径
        file_path = search_file(file, problem)
        # 断言文件存在
        assert file_path is not None
        # 读取文件内容作为代码
        code = open(file_path, "r").read()
    else:
        # 输入为代码字符串，直接使用
        code = file

    # 如果未指定函数名，从文件名中提取（去除路径和扩展名）
    if function_name is None:
        function_name = file.split(os.sep)[-1].split(".")[0]

    # 执行代码，将函数加载到全局变量
    exec(code, globals())
    # 断言函数已加载
    assert function_name in globals()
    # 返回函数对象
    return eval(function_name)

# 定义load_framework_description函数，从组件代码中提取解决方案和操作符的框架描述（仅保留关键方法）
def load_framework_description(component_code: str) -> tuple[str, str]:
    """ Load framework description for the problem from source code, including solution design and operators design."""
    # 内部函数get_method_source，将方法节点转换为带缩进的源代码
    def get_method_source(method_node):
        """Convert the method node to source component_code, ensuring correct indentation."""
        # 将方法节点反解析为字符串并按行分割
        source_lines = ast.unparse(method_node).split('\n')
        # 每行添加4个空格缩进
        indented_source = '\n'.join(['    ' + line for line in source_lines])  # Indent the source component_code
        return indented_source

    # 解析组件代码为抽象语法树（AST）
    tree = ast.parse(component_code)
    solution_str = ""
    operator_str = ""

    # 遍历AST节点，寻找继承自BaseSolution和BaseOperator的类
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            # 处理继承自BaseSolution的类（解决方案类）
            if any(base.id == 'BaseSolution' for base in node.bases if isinstance(base, ast.Name)):
                # 仅提取__init__方法
                for item in node.body:
                    if isinstance(item, ast.FunctionDef) and item.name == '__init__':
                        solution_str += f"class {node.name}:\n"
                        # 添加类的文档字符串（如果有）
                        solution_str += f"    \"\"\"{ast.get_docstring(node)}\"\"\"\n" if ast.get_docstring(node) else ""
                        # 添加__init__方法的源代码
                        solution_str += get_method_source(item) + "\n"
            # 处理继承自BaseOperator的类（操作符类）
            elif any(base.id == 'BaseOperator' for base in node.bases if isinstance(base, ast.Name)):
                operator_str += f"class {node.name}(BaseOperator):\n"
                # 添加类的文档字符串（如果有）
                operator_str += f"    \"\"\"{ast.get_docstring(node)}\"\"\"\n" if ast.get_docstring(node) else ""
                # 提取__init__和run方法
                for item in node.body:
                    if isinstance(item, ast.FunctionDef) and item.name in ['__init__', 'run']:
                        operator_str += get_method_source(item) + "\n"

    # 返回处理后的解决方案和操作符的字符串描述（去除首尾空白）
    return solution_str.strip(), operator_str.strip()

# 定义extract_function_with_docstring函数，从代码字符串中提取指定函数及其文档字符串
def extract_function_with_docstring(code_str, function_name):
    # 正则表达式匹配函数定义及后续的文档字符串（三引号包裹）
    pattern = rf"def {function_name}\(.*?\) -> .*?:\s+\"\"\"(.*?)\"\"\""
    # 搜索匹配内容
    match = re.search(pattern, code_str, re.DOTALL)
    if match:
        return match.group(0)
    else:
        return None

# 定义filter_dict_to_str函数，将字典列表转换为字符串，可指定内容长度阈值过滤长内容
def filter_dict_to_str(dicts: list[dict], content_threshold: int=None) -> str:
    # 如果输入是单个字典，转换为列表
    if isinstance(dicts, dict):
        dicts = [dicts]
    # 合并所有字典为一个总字典
    total_dict = {k: v for d in dicts for k, v in d.items()}
    strs = []
    # 遍历键值对
    for key, value in total_dict.items():
        # 跳过可调用对象（如函数）
        if callable(value):
            continue
        # 将numpy数组转换为列表
        if isinstance(value, np.ndarray):
            value = value.tolist()
        # 格式化键值对字符串（含换行的内容单独处理）
        if "\n" in str(value):
            key_value_str = str(key) + ":\n" + str(value)
        else:
            key_value_str = str(key) + ":" + str(value)
        # 根据阈值过滤（如果指定了阈值且内容长度超过阈值则跳过）
        if content_threshold is None or len(key_value_str) <= content_threshold:
            strs.append(key_value_str)
    # 用换行连接所有键值对字符串并返回
    return "\n".join(strs)

# 定义find_key_value函数，在嵌套字典中查找指定键的值
def find_key_value(source_dict: dict, key: object) -> object:
    # 先在当前字典中查找
    if key in source_dict:
        return source_dict[key]
    else:
        # 递归在嵌套字典中查找
        for k, v in source_dict.items():
            if isinstance(v, dict):
                if key in v:
                    return v[key]
    # 未找到返回None
    return None

# 定义extract_function_with_short_docstring函数，提取函数定义及简短文档字符串（到Args部分为止）
def extract_function_with_short_docstring(code_str, function_name):
    # 正则表达式匹配函数定义及到Args前的文档字符串
    pattern = rf"def {function_name}\(.*?\) -> .*?:\s+\"\"\"(.*?).*Args"
    match = re.search(pattern, code_str, re.DOTALL)
    if match:
        string = match.group(0)
        # 提取函数名（分割括号前的部分）
        function_name = string.split("(")[0].strip()
        # 提取参数（algorithm_data到**kwargs之间的部分）
        parameters = string.split("algorithm_data: dict")[1].split(", **kwargs")[0].strip()
        # 处理参数前可能的逗号和空格
        if parameters[:2] == ", ":
            parameters = parameters[2:]
        # 提取文档字符串中Args前的介绍部分，并去除多余空格
        introduction = string.split("\"\"\"")[1].split("Args")[0].strip()
        introduction = re.sub(r'\s+', ' ', introduction)
        # 返回格式化的函数信息（函数名(参数): 介绍）
        return f"{function_name}({parameters}): {introduction}"
    else:
        return None

# 定义parse_paper_to_dict函数，将LaTeX论文内容解析为按章节层级组织的字典
def parse_paper_to_dict(content: str, level=0):
    # 根据层级定义匹配章节的正则表达式（section、subsection、subsubsection）
    if level == 0:
        pattern = r'\\section\{(.*?)\}(.*?)((?=\\section)|\Z)'
    elif level == 1:
        pattern = r'\\subsection\{(.*?)\}(.*?)((?=\\subsection)|(?=\\section)|\Z)'
    elif level == 2:
        pattern = r'\\subsubsection\{(.*?)\}(.*?)((?=\\subsubsection)|(?=\\subsection)|(?=\\section)|\Z)'
    else:
        raise ValueError("Unsupported section level")
    # 查找所有匹配的章节
    sections = re.findall(pattern, content, re.DOTALL)
    section_dict = {}
    # 遍历章节，递归解析子章节
    for title, body, _ in sections:
        body = body.strip()
        if level < 2:
            # 递归解析下一级章节
            sub_dict = parse_paper_to_dict(body, level + 1)
            if sub_dict:
                section_dict[title] = sub_dict
            else:
                section_dict[title] = body
        else:
            section_dict[title] = body
    # 处理顶层的摘要和标题
    if level == 0:
        if "\\begin{abstract}" in content:
            section_dict["Abstract"] = content.split("\\begin{abstract}")[-1].split("\\end{abstract}")[0]
        if "\\begin{Abstract}" in content:
            section_dict["Abstract"] = content.split("\\begin{Abstract}")[-1].split("\\end{Abstract}")[0]
        if "\\title{" in content:
            section_dict["Title"] = content.split("\\title{")[-1].split("}")[0]
    return dict(section_dict)

# 定义replace_strings_in_dict函数，将字典中所有字符串值替换为指定值（用于简化大字典的显示）
def replace_strings_in_dict(source_dict: dict, replace_value: str="...") -> dict:
    # 遍历字典键值对
    for key in source_dict:
        # 如果值是字符串，替换为指定值
        if isinstance(source_dict[key], str):
            source_dict[key] = replace_value 
        # 如果值是字典，递归处理
        elif isinstance(source_dict[key], dict):
            source_dict[key] = replace_strings_in_dict(source_dict[key])
    return source_dict

# 定义search_file函数，在指定问题的相关目录中搜索文件并返回路径
def search_file(file_name: str, problem: str="base") -> str:
    # 内部函数find_file_in_folder，在指定文件夹中递归查找文件
    def find_file_in_folder(folder_path, file_name):
        return next((os.path.join(root, file_name) for root, dirs, files in os.walk(folder_path) if file_name in files or file_name in dirs), None)

    # 先检查文件是否在当前路径存在
    if os.path.exists(file_name):
        return file_name

    # 在问题的源代码目录中查找
    file_path = find_file_in_folder(os.path.join("src", "problems", problem), file_name)
    if file_path:
        return file_path

    # 确定输出目录（优先使用环境变量AMLT_DATA_DIR，否则使用"output"）
    if os.getenv("AMLT_DATA_DIR"):
        output_dir = os.getenv("AMLT_DATA_DIR")
    else:
        output_dir = "output"

    # 在问题的输出数据目录中查找
    file_path = find_file_in_folder(os.path.join(output_dir, problem, "data"), file_name)
    if file_path:
        return file_path

    # 在问题的启发式算法目录中查找
    file_path = find_file_in_folder(os.path.join(output_dir, problem, "heuristics"), file_name)
    if file_path:
        return file_path

    # 在问题的输出根目录中查找
    file_path = find_file_in_folder(os.path.join(output_dir, problem), file_name)
    if file_path:
        return file_path
    # 未找到返回None
    return None

# 定义df_to_str函数，将DataFrame转换为制表符分隔的字符串
def df_to_str(df: pd.DataFrame) -> str:
    return df.to_csv(sep="\t", index=False).replace("\r\n", "\n").strip()

# 定义str_to_df函数，将制表符分隔的字符串转换为DataFrame
def str_to_df(string: str) -> pd.DataFrame:
    return pd.read_csv(io.StringIO(string), sep="\t")

# 定义compress_numbers函数，将字符串中超过两位小数的浮点数格式化为两位小数
def compress_numbers(s):
    # 内部函数format_float，处理匹配到的浮点数
    def format_float(match):
        number = match.group()
        # 如果是包含小数点且小数部分超过两位的数，保留两位小数
        if '.' in number and len(number.split('.')[1]) > 2:
            return "{:.2f}".format(float(number))
        return number
    # 正则表达式匹配小数部分超过两位的浮点数
    return re.sub(r'\d+\.\d{3,}', format_float, s)

# 定义sanitize_function_name函数，将名称处理为合法的函数名（包含哈希后缀避免重复）
def sanitize_function_name(name: str, id_str: str="None"):
    # 使用正则表达式将驼峰命名转换为下划线命名
    s1 = re.sub('(.)([A-Z][a-z]+)', r"\1_\2", name)
    sanitized_name = re.sub('([a-z0-9])([A-Z])', r"\1_\2", s1).lower()

    # 将空格替换为下划线，处理连续下划线
    sanitized_name = sanitized_name.replace(" ", "_").replace("__", "_")

    # 移除无效字符（只保留字母、数字和下划线）
    sanitized_name = "".join(char for char in sanitized_name if char.isalnum() or char == '_')

    # 确保函数名不以数字开头（如果以数字开头，添加下划线前缀）
    if sanitized_name and sanitized_name[0].isdigit():
        sanitized_name = "_" + sanitized_name

    # 生成id_str的哈希值前4位作为后缀，避免名称重复
    suffix_str = hashlib.sha256(id_str.encode()).hexdigest()[:4]
    sanitized_name = sanitized_name + "_" + suffix_str

    return sanitized_name