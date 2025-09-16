from src.pipeline.hyper_heuristics.random import RandomHyperHeuristic
from src.problems.base.env import BaseEnv
from src.util.util import load_function

def run_random_hh(
        env: BaseEnv,  # 直接使用环境对象，无需序列化
        heuristic_pool: list[str],
        problem: str,
        iterations_scale_factor: float,
        best_result: list[float]  # 使用列表存储最佳结果（模拟引用传递）
) -> float:
    # 实例化随机超启发式算法
    random_hh = RandomHyperHeuristic(heuristic_pool, problem, iterations_scale_factor)
    # 运行算法
    complete_and_valid_solution = random_hh.run(env)

    if complete_and_valid_solution:
        # 检查是否为最佳结果
        if best_result[0] == float('-inf') or env.compare(env.key_value, best_result[0]) >= 0:
            best_result[0] = env.key_value
            env.dump_result(result_file=f"best_result_{env.key_value}.txt")
        return env.key_value
    else:
        return None

def evaluate_heuristic(
        env: BaseEnv,  # 直接使用环境对象
        heuristic_name: str,
        heuristic_pool: list[str],
        problem: str,
        iterations_scale_factor: float,
        steps_per_selection: int,
        rollout_budget: int,
        best_result: list[float]  # 使用列表存储最佳结果
) -> tuple[str, list[float]]:
    # 加载启发式算法
    heuristic = load_function(heuristic_name, problem)
    
    # 执行指定步骤的启发式算法
    operators = []
    for _ in range(steps_per_selection):
        operators.append(env.run_heuristic(heuristic))
    
    # 保存当前环境状态的副本（用于多次评估）
    # 注意：这里假设env有copy方法，实际使用时可能需要调整
    base_env = env.copy()
    
    # 执行滚动评估（单线程）
    results = []
    for _ in range(rollout_budget):
        # 每次评估都使用环境的副本，避免相互影响
        env_copy = base_env.copy()
        result = run_random_hh(
            env_copy,
            heuristic_pool,
            problem,
            iterations_scale_factor,
            best_result
        )
        if result:
            results.append(result)
    
    return heuristic_name, results

def tts_bon(
        env: BaseEnv,
        candidate_heuristics: list[str],
        heuristic_pool: list[str],
        problem: str,
        iterations_scale_factor: float,
        steps_per_selection: int,
        rollout_budget: int,
) -> str:
    # 边界条件处理
    if rollout_budget == 0 or len(candidate_heuristics) == 1:
        return candidate_heuristics[0]
    
    # 存储最佳结果（使用列表实现类似引用的效果）
    best_result = [float('-inf')]
    
    # 存储所有候选算法的评估结果
    evaluation_results = []
    
    # 逐个评估候选算法（单线程）
    for heuristic in candidate_heuristics:
        # 每次评估都使用环境的副本，避免相互影响
        env_copy = env.copy()
        heuristic_name, results = evaluate_heuristic(
            env_copy,
            heuristic,
            heuristic_pool,
            problem,
            iterations_scale_factor,
            steps_per_selection,
            rollout_budget,
            best_result
        )
        evaluation_results.append((heuristic_name, results))
    
    # 找出最佳启发式算法
    best_average_score = None
    best_heuristic_name = candidate_heuristics[0]
    
    for heuristic_name, results in evaluation_results:
        # 计算平均分数
        if len(results) > 0:
            average_score = sum(results) / len(results)
            
            # 更新最佳算法
            if best_average_score is None or env.compare(average_score, best_average_score) > 0:
                best_average_score = average_score
                best_heuristic_name = heuristic_name
    
    return best_heuristic_name
