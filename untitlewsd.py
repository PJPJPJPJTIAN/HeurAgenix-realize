
print(f"('===============Evolution {basic_heuristic_name} on {evolution_data}('===============")  # 打印进化信息

prompt_dict = self.llm_client.load_background(self.problem, 
                                              "background_with_code") 
# 加载问题背景信息到提示字典
prompt_dict["all_heuristic_docs"] = all_heuristic_docs
# 将所有启发式文档添加到提示字典
self.load_function_code(basic_heuristic_file, prompt_dict) 
# 加载基础启发式的代码到提示字典

# 识别瓶颈操作
bottlenecks = self.identity_bottlenecks(  
    # 调用识别瓶颈方法，找出导致基础启发式性能不佳的操作
    prompt_dict=prompt_dict,  # 提示字典
    env=env,  # 环境实例
    positive_result=positive_result,  # 正样本结果
    negative_result=negative_result  # 负样本结果
)

for bottleneck_index, (bottleneck_operation_id, 
                       proposed_operation, reason) in enumerate(bottlenecks):  # 遍历每个瓶颈
    # 提出改进建议并生成进化后的启发式
    print('===============一共有多少个bottleneck_index===============',len(bottlenecks))
    print('===============第一个===============',bottleneck_index)
    basic_heuristic_result = self.validation(self.validation_cases,
                                             basic_heuristic_file)  # 验证基础启发式在验证集上的表现

    print('basic_heuristic_resultbasic_heuristic_resultbasic_heuristic_result==========',basic_heuristic_result)
    # basic_heuristic_resultbasic_heuristic_resultbasic_heuristic_result========== [2309.0, 3119.0, 25407.0]
    suggestion_name = f"suggestion_{bottleneck_index}"  # 定义建议名称
    # 返回新启发式文件路径、建议和验证结果
    suggested_heuristic_file, suggestion, suggested_result = self.raise_suggestion( 
        # 调用提出建议方法，生成改进后的启发式
        prompt_dict=prompt_dict,  # 提示字典
        env=env,  # 环境实例
        bottleneck_operation_id=bottleneck_operation_id,  # 瓶颈操作ID
        proposed_operation=proposed_operation,  # 建议的操作
        reason=reason,  # 建议的理由
        suggestion_name=suggestion_name,  # 建议名称
        smoke_test=smoke_test  # 是否快速测试
    )
    # 返回新启发式文件路径、建议和验证结果
    if suggested_heuristic_file:  # 如果成功生成了改进的启发式文件
        output_heuristic_name = suggested_heuristic_file.split(os.sep)[-1].split(".")[0] 
        # 提取输出启发式名称
        self.llm_client.dump(f"{basic_heuristic_name}_to_{output_heuristic_name}")  
        # 保存LLM交互记录

        suggested_improvement = sum(self.get_improvement(
                            env, basic_heuristic_result, suggested_result)) / len(basic_heuristic_result) 
        # 计算平均改进值
        print(f"========Improvement for {suggested_heuristic_file}: {suggested_improvement}")  # 打印改进信息
        # 第一次添加
        refined_heuristic_benchmarks.append([suggested_heuristic_file, suggested_improvement]) 
        # 将改进的启发式及其改进值添加到列表
        
        # 进一步微调进化后的启发式
        previous_heuristic_name = basic_heuristic_name  # 记录上一个启发式名称
        previous_heuristic_result = basic_heuristic_result  # 记录上一个启发式结果
        last_heuristic_name = suggested_heuristic_file.split(os.sep)[-1].split(".")[0]  
        # 记录当前启发式名称
        last_heuristic_result = suggested_result  # 记录当前启发式结果
        last_suggestion = suggestion  # 记录上一个建议

        # 在 for refine_index in range(max_refinement_round) 循环中，
        # 涉及的三个启发式（basic_heuristic、previous_heuristic、last_heuristic）
        # 是迭代优化过程中的三个关键节点，分别代表基准、上一轮结果和当前结果
#  三个启发式的定义
# basic_heuristic：
# 本轮进化的初始基准启发式（即 evolution_single 方法的输入 basic_heuristic_file），是整个优化链条的起点，
# 其性能作为评估后续改进的参照基准。
# previous_heuristic：
# 上一次精炼迭代中生成的启发式，是当前迭代的 “前身”。在循环初始时，它被初始化为 basic_heuristic，
# 之后每轮迭代会更新为上一轮的 last_heuristic。
# last_heuristic：
# 当前最新生成的启发式（本轮迭代的起点），是待进一步精炼的对象。在循环初始时，它是基于瓶颈改进生成的 suggested_heuristic，
# 之后每轮迭代会更新为新生成的精炼启发式。
        
        for refine_index in range(max_refinement_round):  # 循环执行最大优化轮次
            suggestion_name = f"suggestion_{bottleneck_index}_refine_{refine_index}"  
            # 定义优化建议名称
            refined_heuristic_file, suggestion, refined_result = self.refine_heuristic(  
                # 调用优化启发式方法，进一步改进启发式
                prompt_dict=prompt_dict,  # 提示字典
                env=env,  # 环境实例
                basic_heuristic_name=basic_heuristic_name,  # 基础启发式名称
                basic_heuristic_result=basic_heuristic_result,  # 基础启发式结果
                previous_heuristic_name=previous_heuristic_name,  # 上一个启发式名称
                previous_heuristic_result=previous_heuristic_result,  # 上一个启发式结果
                last_heuristic_name=last_heuristic_name,  # 当前启发式名称
                last_heuristic_result=last_heuristic_result,  # 当前启发式结果
                last_suggestion=last_suggestion,  # 上一个建议
                suggestion_name=suggestion_name,  # 建议名称
                smoke_test=smoke_test  # 是否快速测试
            )
            if refined_result is None or None in refined_result:
                print("Error and skip")
                continue
            if refined_heuristic_file:  # 如果成功生成了优化后的启发式文件
                output_heuristic_name = refined_heuristic_file.split(os.sep)[-1].split(".")[0]  
                # 提取输出启发式名称
                self.llm_client.dump(f"{last_heuristic_name}_to_{output_heuristic_name}") 
                # 保存LLM交互记录
                refined_improvement = sum(self.get_improvement(
                                    env, basic_heuristic_result, refined_result)) / len(basic_heuristic_result)  
                # refined_improvement为在validation数据上的计算平均改进值
                print(f"Improvement for {refined_heuristic_file}: {refined_improvement}") 
                # 打印改进信息
                refined_heuristic_benchmarks.append([refined_heuristic_file, refined_improvement]) 
                # 将优化的启发式及其改进值添加到列表
                if suggestion is None:  # 如果没有新的建议
                    break  # 跳出优化循环
                previous_heuristic_name = last_heuristic_name  # 更新上一个启发式名称
                previous_heuristic_result = last_heuristic_result  # 更新上一个启发式结果
                last_suggestion = suggestion  # 更新上一个建议
                last_heuristic_name = refined_heuristic_file.split(os.sep)[-1].split(".")[0]  
                # 更新当前启发式名称
                last_heuristic_result = refined_result  # 更新当前启发式结果

return refined_heuristic_benchmarks  # 返回优化后的启发式基准列表


# ==========================================================================================================================

def raise_suggestion(  # 定义提出建议的方法，用于根据瓶颈生成改进建议并生成新的启发式
        self,
        prompt_dict: dict,  # 提示字典
        env: BaseEnv,  # 环境实例
        bottleneck_operation_id: int,  # 瓶颈操作ID
        proposed_operation: str,  # 建议的操作
        reason: str,  # 建议的理由
        suggestion_name: str,  # 建议名称
        smoke_test: bool,  # 是否快速测试
) -> tuple[str, str]:  # 返回新的启发式文件路径、建议和其验证结果
    env.reset()  # 重置环境
    self.llm_client.load_chat("bottleneck_operations")  # 加载之前识别瓶颈的聊天记录
    prompt_dict["bottleneck_operation_id"] = bottleneck_operation_id  # 将瓶颈操作ID添加到提示字典
    prompt_dict["proposed_operation"] = proposed_operation  # 将建议的操作添加到提示字典
    prompt_dict["reason"] = reason  # 将建议的理由添加到提示字典
    # 将负样本轨迹字符串转换为DataFrame
    negative_trajectory_df = pd.read_csv(StringIO(prompt_dict["negative_trajectory"]), sep="\t")  
    # 获取瓶颈操作的具体内容
    bottleneck_operation = list(negative_trajectory_df[negative_trajectory_df["operation_id"] == bottleneck_operation_id]["operator"])[0]  
    # 遍历瓶颈操作之前的所有操作
    for previous_operation in negative_trajectory_df[negative_trajectory_df["operation_id"] < bottleneck_operation_id]["operator"]:  
        # 在环境中执行这些操作，重现瓶颈前的状态
        env.run_operator(eval(previous_operation))  
    prompt_dict["bottleneck_operation"] = bottleneck_operation  # 将瓶颈操作添加到提示字典
    # 将瓶颈前的解决方案添加到提示字典
    prompt_dict["solution_before_bottleneck"] = str(env.current_solution)  
    # 将瓶颈前的解决方案问题状态添加到提示字典
    prompt_dict["solution_problem_state_before_bottleneck"] = filter_dict_to_str(self.get_solution_problem_state(env.instance_data, env.current_solution))  

    # 调用LLM获取改进建议
    self.llm_client.load("extract_suggestion", prompt_dict)  # 加载提取建议的提示模板和参数
    response = self.llm_client.chat()  # 与LLM交互，获取响应
    suggestion = extract(response, key="suggestion")  # 从响应中提取建议
    if suggestion:  # 如果获取到建议
        self.llm_client.dump(suggestion_name)  # 保存LLM交互记录
        # 根据建议生成新的启发式代码
        heuristic_name = prompt_dict["heuristic_name"]  # 获取启发式名称
        origin_function_name = prompt_dict["function_name"]  # 获取原始函数名称
        prompt_dict["suggestion"] = suggestion  # 将建议添加到提示字典
        description = f"Now, based on these suggestions:\n{suggestion}\nUpdate the {origin_function_name}."  # 生成新启发式的描述
        env_summarize = prompt_dict["env_summarize"]  # 获取环境摘要信息
        # 放心这里在prompt_dict = self.llm_client.load_background(self.problem,"background_with_code") 中已经加入了
        more_prompt_dict = {"problem_state_introduction": prompt_dict['problem_state_introduction']}
        output_heuristic_file = HeuristicGenerator(self.llm_client, self.problem).generate(heuristic_name, description, env_summarize, smoke_test, more_prompt_dict = more_prompt_dict)  
        # 调用启发式生成器生成新的启发式文件
        if output_heuristic_file:  # 如果成功生成新的启发式文件
            # 在验证集上验证新启发式
            suggested_result = self.validation(self.validation_cases, output_heuristic_file)  
            # 返回新启发式文件路径、建议和验证结果
            return output_heuristic_file, suggestion, suggested_result  
    return None, None, None  # 如果没有生成有效的新启发式，返回None

def validation(  # 定义验证方法，用于在验证集上评估启发式的性能
        self,
        validation_cases: list[str],  # 验证数据集路径列表
        heuristic_file: str  # 要验证的启发式文件路径
    ) -> list[float]:  # 返回每个验证数据上的结果值列表
    validation_results = []  # 初始化验证结果列表
    heuristic_name = heuristic_file.split(os.sep)[-1].split(".py")[0]  # 从文件路径中提取启发式名称
    for data_name in validation_cases:  # 遍历每个验证数据
        env = Env(data_name=data_name)  # 创建环境实例，加载验证数据
        env.reset(heuristic_name)  # 重置环境，设置启发式名称
        hyper_heuristic = SingleHyperHeuristic(heuristic_file, problem=self.problem)  # 创建单启发式执行器实例
        is_complete_valid_solution = hyper_heuristic.run(env)
        result = env.key_value if is_complete_valid_solution else None
        # if 成功，env.is_complete_solution, and env.is_valid_solution.返回 True = is_complete_valid_solution
        validation_results.append(result)
        # result 就是函数的目标函数值
    return validation_results

def get_improvement(self, env:BaseEnv, baselines: list[float], results: list[float]) -> float:
    improvements = [round(env.compare(results[index], baselines[index]) / baselines[index], 2) for index in range(len(baselines))]
    return improvements