import concurrent  # 导入concurrent模块，用于并发执行任务
import dill  # 导入dill模块，用于序列化和反序列化Python对象（支持比pickle更多类型）
import multiprocessing  # 导入multiprocessing模块，用于多进程处理
import multiprocessing.managers  # 导入multiprocessing.managers，用于创建跨进程共享的管理器
from src.pipeline.hyper_heuristics.random import RandomHyperHeuristic  # 从指定路径导入随机超启发式类
from src.problems.base.env import BaseEnv  # 从指定路径导入基础环境类
from src.util.util import load_function  # 从工具模块导入加载函数的工具函数

dill.settings['recurse'] = True  # 配置dill的序列化设置，允许递归序列化对象

def run_random_hh(  # 定义一个运行随机超启发式的函数
        env_serialized: bytes,  # 序列化后的环境对象（字节类型）
        heuristic_pool: list[str],  # 启发式算法池（包含启发式名称的列表）
        problem: str,  # 问题名称（字符串）
        iterations_scale_factor: float,  # 迭代缩放因子（控制迭代次数的比例）
        best_result_proxy: multiprocessing.managers.ValueProxy,  # 跨进程共享的最佳结果代理
) -> float:  # 函数返回值为浮点型（问题的关键值结果）
    random_hh = RandomHyperHeuristic(heuristic_pool, problem, iterations_scale_factor)  # 实例化随机超启发式对象
    env = dill.loads(env_serialized)  # 反序列化环境对象，恢复环境状态
    complete_and_valid_solution = random_hh.run(env)  # 运行随机超启发式，返回是否得到完整且有效的解

    if complete_and_valid_solution:  # 如果得到有效解
        # 若当前最佳结果为负无穷，或当前环境的关键值优于最佳结果，则更新最佳结果
        if best_result_proxy.value == float('-inf') or env.compare(env.key_value, best_result_proxy.value) >= 0:
            best_result_proxy.value = env.key_value  # 更新共享的最佳结果值
            env.dump_result(result_file=f"best_result_{env.key_value}.txt")  # 将结果保存到文件
        return env.key_value  # 返回当前环境的关键值
    else:  # 若未得到有效解
        return None  # 返回None

def evaluate_heuristic(  # 定义一个评估启发式算法的函数
        env_serialized: bytes,  # 序列化后的环境对象
        heuristic_name: str,  # 待评估的启发式算法名称
        heuristic_pool: list[str],  # 启发式算法池
        problem: str,  # 问题名称
        iterations_scale_factor: float,  # 迭代缩放因子
        steps_per_selection: int,  # 每次选择后执行的步数
        rollout_budget: int,  # 滚动预算（用于评估的采样次数）
        best_result_proxy: multiprocessing.managers.ValueProxy,  # 共享的最佳结果代理
) -> tuple[float, str, bytes]:  # 返回值为（启发式名称，结果列表）的元组
    env = dill.loads(env_serialized)  # 反序列化环境，恢复初始状态
    heuristic = load_function(heuristic_name, problem)  # 加载指定名称的启发式函数
    operators = []  # 初始化操作列表，用于存储启发式生成的操作
    for _ in range(steps_per_selection):  # 循环执行指定步数
        operators.append(env.run_heuristic(heuristic))  # 运行启发式算法，获取操作并添加到列表
    after_step_env_serialized = dill.dumps(env)  # 序列化执行后的环境状态
    # 使用MCTS（蒙特卡洛树搜索）评估启发式性能
    results = []  # 初始化结果列表，存储每次滚动的结果
    future_results = []  # 初始化未来结果列表，存储并发任务的未来对象
    with concurrent.futures.ThreadPoolExecutor() as executor:  # 创建线程池执行器
        for _ in range(rollout_budget):  # 循环执行滚动预算次数
            future_results.append(executor.submit(  # 提交run_random_hh任务到线程池
                run_random_hh,  # 目标函数
                after_step_env_serialized,  # 执行后的环境序列化对象
                heuristic_pool,  # 启发式池
                problem,  # 问题名称
                iterations_scale_factor,  # 迭代缩放因子
                best_result_proxy,  # 最佳结果代理
                ))
    for future in concurrent.futures.as_completed(future_results):  # 遍历完成的并发任务
        result = future.result()  # 获取任务结果
        if result:  # 若结果有效
            results.append(result)  # 添加到结果列表
    return heuristic_name, results  # 返回启发式名称和对应的结果列表

def tts_bon(  # 定义tts_bon函数（可能是一种启发式选择策略）
        env: BaseEnv,  # 环境对象（BaseEnv实例）
        candidate_heuristics: list[str],  # 候选启发式算法列表
        heuristic_pool: list[str],  # 启发式算法池
        problem: str,  # 问题名称
        iterations_scale_factor: float,  # 迭代缩放因子
        steps_per_selection: int,  # 每次选择后执行的步数
        rollout_budget: int,  # 滚动预算
) -> tuple[str, bytes]:  # 返回值为（最佳启发式名称，序列化环境）的元组
    if rollout_budget == 0 or len(candidate_heuristics) == 1:  # 若滚动预算为0或只有一个候选启发式
        return candidate_heuristics[0]  # 直接返回该候选启发式
    manager = multiprocessing.Manager()  # 创建多进程管理器，用于共享状态
    best_result_proxy = manager.Value('d', float('-inf'))  # 创建共享的最佳结果代理（初始为负无穷）
    env_serialized = dill.dumps(env)  # 序列化当前环境状态
    futures = []  # 初始化未来任务列表
    for heuristic in candidate_heuristics:  # 遍历每个候选启发式
        with concurrent.futures.ProcessPoolExecutor() as executor:  # 创建进程池执行器
            futures.append(executor.submit(  # 提交评估任务到进程池
                evaluate_heuristic,  # 目标函数
                env_serialized,  # 环境序列化对象
                heuristic,  # 当前候选启发式名称
                heuristic_pool,  # 启发式池
                problem,  # 问题名称
                iterations_scale_factor,  # 迭代缩放因子
                steps_per_selection,  # 每次选择的步数
                rollout_budget,  # 滚动预算
                best_result_proxy  # 最佳结果代理
            ))

    best_average_score = None  # 初始化最佳平均分数为None
    for future in concurrent.futures.as_completed(futures):  # 遍历完成的评估任务
        heuristic_name, results  = future.result()  # 获取任务返回的启发式名称和结果列表
        # 计算平均分数（若结果列表非空）
        average_score = None if len(results) <= 0 else sum(results) / len(results)
        best_heuristic_name = candidate_heuristics[0]  # 初始化最佳启发式名称为第一个候选
        if average_score is not None:  # 若平均分数有效
            # 若当前最佳平均分数为空，或当前平均分数更优（通过环境的compare方法判断）
            if best_average_score is None or env.compare(average_score, best_average_score) > 0:
                best_heuristic_name = heuristic_name  # 更新最佳启发式名称
                best_average_score = average_score  # 更新最佳平均分数
    return best_heuristic_name  # 返回最佳启发式名称