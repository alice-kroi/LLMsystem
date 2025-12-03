import logging
import os
import time
import json
from prompt import get_prompt, CONTRACT_GENERATION_PROMPT, CONTRACT_REVIEW_PROMPT, CUSTOMER_SERVICE_PROMPT
from Agent import Agent

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('agent_test')


def test_llm_creation():
    """
    测试LLM创建功能，包括不同提供商的测试
    """
    logger.info("开始测试LLM创建功能")
    
    # 测试默认LLM创建（智谱AI）
    agent = Agent()
    result = agent.create_llm()
    assert result, "默认LLM创建失败"
    assert agent.llm is not None, "LLM实例未创建"
    logger.info("智谱AI LLM创建测试通过")
    
    # 测试OpenAI兼容接口
    # 注意：这里可能需要配置有效的API密钥才能通过测试
    # 为了避免测试失败，这里仅测试函数调用，不验证具体实例
    agent2 = Agent()
    try:
        agent2.create_llm(provider="openai", model_name="gpt-3.5-turbo")
        logger.info("尝试创建OpenAI兼容LLM完成")
    except Exception as e:
        logger.warning(f"OpenAI LLM创建可能失败（需要有效API密钥）: {e}")
    
    # 测试不支持的提供商
    agent3 = Agent()
    result = agent3.create_llm(provider="unsupported")
    assert not result, "不支持的提供商测试失败"
    logger.info("不支持提供商测试通过")
    
    logger.info("LLM创建功能测试完成")


def test_single_conversation():
    """
    测试单一对话功能
    """
    logger.info("开始测试单一对话功能")
    
    agent = Agent()
    user_input = "你好，请介绍一下你自己"
    prompt = get_prompt(CUSTOMER_SERVICE_PROMPT, user_input)
    
    if prompt:
        result = agent.generate_response(prompt)
        assert result is not None, "生成响应失败"
        assert 'response' in result, "响应中缺少response字段"
        assert 'conversation_id' in result, "响应中缺少conversation_id字段"
        logger.info(f"单一对话测试通过，生成的对话ID: {result['conversation_id']}")
    else:
        logger.error("获取提示词失败")
        assert False, "获取提示词失败"


def test_conversation_memory():
    """
    测试对话记忆功能
    """
    logger.info("开始测试对话记忆功能")
    
    agent = Agent()
    
    # 第一步：进行初始对话
    user_input1 = "我叫张三，今年25岁"
    prompt1 = get_prompt(CUSTOMER_SERVICE_PROMPT, user_input1)
    
    if prompt1:
        result1 = agent.generate_response(prompt1)
        assert result1 is not None, "第一次对话生成响应失败"
        conversation_id = result1['conversation_id']
        
        # 第二步：验证记忆
        user_input2 = "我刚才告诉过你我的名字吗？如果是的，我叫什么？"
        prompt2 = get_prompt(CUSTOMER_SERVICE_PROMPT, user_input2)
        
        if prompt2:
            result2 = agent.generate_response(prompt2, conversation_id=conversation_id)
            assert result2 is not None, "第二次对话生成响应失败"
            # 验证回复中包含记忆信息
            response_lower = result2['response'].lower()
            assert '张三' in result2['response'], "记忆测试失败，AI未记住名字"
            logger.info("对话记忆功能测试通过")
        else:
            logger.error("获取第二个提示词失败")
            assert False, "获取第二个提示词失败"
    else:
        logger.error("获取第一个提示词失败")
        assert False, "获取第一个提示词失败"


def test_multiple_conversations():
    """
    测试多个独立对话的管理
    """
    logger.info("开始测试多对话管理功能")
    
    agent = Agent()
    
    # 创建第一个对话
    user_input1 = "我叫张三，我喜欢编程"
    prompt1 = get_prompt(CUSTOMER_SERVICE_PROMPT, user_input1)
    result1 = agent.generate_response(prompt1)
    assert result1 is not None, "第一个对话创建失败"
    conversation_id1 = result1['conversation_id']
    
    # 创建第二个对话
    user_input2 = "我叫李四，我喜欢音乐"
    prompt2 = get_prompt(CUSTOMER_SERVICE_PROMPT, user_input2)
    result2 = agent.generate_response(prompt2)
    assert result2 is not None, "第二个对话创建失败"
    conversation_id2 = result2['conversation_id']
    
    # 确保两个对话ID不同
    assert conversation_id1 != conversation_id2, "两个对话应该有不同的ID"
    
    # 验证第一个对话的记忆
    verify_input1 = "我叫什么名字？我喜欢什么？"
    verify_prompt1 = get_prompt(CUSTOMER_SERVICE_PROMPT, verify_input1)
    verify_result1 = agent.generate_response(verify_prompt1, conversation_id=conversation_id1)
    assert '张三' in verify_result1['response'] or '编程' in verify_result1['response'], "第一个对话记忆测试失败"
    
    # 验证第二个对话的记忆
    verify_input2 = "我叫什么名字？我喜欢什么？"
    verify_prompt2 = get_prompt(CUSTOMER_SERVICE_PROMPT, verify_input2)
    verify_result2 = agent.generate_response(verify_prompt2, conversation_id=conversation_id2)
    assert '李四' in verify_result2['response'] or '音乐' in verify_result2['response'], "第二个对话记忆测试失败"
    
    logger.info("多对话管理功能测试通过")


def test_memory_persistence():
    """
    测试记忆持久化功能
    """
    logger.info("开始测试记忆持久化功能")
    
    # 第一部分：创建对话并保存记忆
    agent1 = Agent()
    user_input = "这是一条用于测试记忆持久化的消息"
    prompt = get_prompt(CUSTOMER_SERVICE_PROMPT, user_input)
    result = agent1.generate_response(prompt)
    assert result is not None, "生成响应失败"
    conversation_id = result['conversation_id']
    
    # 检查记忆文件是否创建
    memory_file = os.path.join(agent1.memory_dir, f"{conversation_id}.json")
    assert os.path.exists(memory_file), "记忆文件未创建"
    
    # 检查文件内容
    try:
        with open(memory_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        assert 'conversations' in data, "记忆文件格式错误"
        assert len(data['conversations']) > 0, "记忆文件内容为空"
        logger.info("记忆文件创建和内容验证通过")
    except Exception as e:
        logger.error(f"记忆文件验证失败: {e}")
        assert False, "记忆文件验证失败"
    
    # 第二部分：重新创建Agent实例并加载记忆
    agent2 = Agent()
    verify_input = "我刚才说了什么？"
    verify_prompt = get_prompt(CUSTOMER_SERVICE_PROMPT, verify_input)
    verify_result = agent2.generate_response(verify_prompt, conversation_id=conversation_id)
    
    assert verify_result is not None, "加载记忆后生成响应失败"
    # 检查回复是否包含之前的消息
    assert '记忆持久化' in verify_result['response'] or '测试' in verify_result['response'], "记忆加载测试失败"
    
    logger.info("记忆持久化功能测试通过")


def test_error_handling():
    """
    测试错误处理功能
    """
    logger.info("开始测试错误处理功能")
    
    agent = Agent()
    
    # 测试空提示词
    result = agent.generate_response("")
    # 这里可能会返回None或空响应，取决于具体实现
    logger.info("空提示词测试完成")
    
    # 测试无效的conversation_id（非UUID格式）
    try:
        result = agent.generate_response("测试消息", conversation_id="invalid-id")
        # 即使ID无效，系统也应该能够处理
        assert result is not None, "无效ID处理失败"
        logger.info("无效conversation_id测试通过")
    except Exception as e:
        logger.error(f"无效ID测试出错: {e}")
        assert False, "无效ID测试失败"
    
    logger.info("错误处理功能测试完成")


def test_contract_memory_flow():
    """
    测试生成雇佣合同并验证对话记忆功能
    """
    logger.info("开始测试雇佣合同生成和对话记忆功能")
    
    # 初始化Agent
    agent = Agent()
    
    # 测试1：生成雇佣合同
    logger.info("测试1：生成雇佣合同")
    user_input1 = "请创建一份雇佣合同，甲方是ABC科技有限公司，乙方是张三，职位是软件工程师，月薪25000元，试用期3个月。"
    prompt1 = get_prompt(CONTRACT_GENERATION_PROMPT, user_input1)
    
    if prompt1:
        # 不指定conversation_id，系统会自动生成新的
        result1 = agent.generate_response(prompt1)
        
        if result1:
            conversation_id = result1['conversation_id']
            logger.info(f"生成的对话ID: {conversation_id}")
            logger.info("雇佣合同生成结果:")
            print("="*50)
            print(result1['response'])
            print("="*50)
            print()
            
            # 测试2：询问之前问的是什么
            logger.info("测试2：询问之前问的是什么")
            user_input2 = "我之前问的是什么？"
            prompt2 = get_prompt(CUSTOMER_SERVICE_PROMPT, user_input2)
            
            if prompt2:
                # 使用相同的conversation_id来保持对话上下文
                result2 = agent.generate_response(prompt2, conversation_id=conversation_id)
                
                if result2:
                    logger.info("关于之前问题的回答:")
                    print("="*50)
                    print(result2['response'])
                    print("="*50)
                    print()
                    
                    # 测试3：询问之前生成的合同是什么
                    logger.info("测试3：询问之前生成的合同是什么")
                    user_input3 = "我之前生成的合同是什么？"
                    prompt3 = get_prompt(CUSTOMER_SERVICE_PROMPT, user_input3)
                    
                    if prompt3:
                        # 继续使用相同的conversation_id
                        result3 = agent.generate_response(prompt3, conversation_id=conversation_id)
                        
                        if result3:
                            logger.info("关于之前合同的回答:")
                            print("="*50)
                            print(result3['response'])
                            print("="*50)
                            logger.info("所有测试完成")
                        else:
                            logger.error("测试3生成响应失败")
                else:
                    logger.error("测试2生成响应失败")
        else:
            logger.error("测试1生成响应失败")


def run_all_tests():
    """
    运行所有测试用例
    """
    logger.info("开始运行所有测试用例")
    
    tests = [
        test_llm_creation,
        test_single_conversation,
        test_conversation_memory,
        test_multiple_conversations,
        test_memory_persistence,
        test_error_handling,
        test_contract_memory_flow
    ]
    
    results = {}
    
    for test in tests:
        test_name = test.__name__
        logger.info(f"正在运行测试: {test_name}")
        try:
            start_time = time.time()
            test()
            end_time = time.time()
            results[test_name] = {
                'status': 'passed',
                'time': f"{end_time - start_time:.2f}s"
            }
            logger.info(f"测试 {test_name} 成功，耗时: {end_time - start_time:.2f}s")
        except Exception as e:
            results[test_name] = {
                'status': 'failed',
                'error': str(e)
            }
            logger.error(f"测试 {test_name} 失败: {e}")
        # 测试间隔
        time.sleep(1)
    
    # 打印测试结果摘要
    logger.info("所有测试完成，结果摘要:")
    print("\n" + "="*60)
    print("测试结果摘要")
    print("="*60)
    
    passed_count = sum(1 for r in results.values() if r['status'] == 'passed')
    failed_count = len(results) - passed_count
    
    for test_name, result in results.items():
        status = "✅ 通过" if result['status'] == 'passed' else f"❌ 失败: {result['error']}"
        time_info = f" (耗时: {result['time']})" if 'time' in result else ""
        print(f"{test_name}: {status}{time_info}")
    
    print("="*60)
    print(f"总计: {passed_count} 通过, {failed_count} 失败")
    print("="*60)


if __name__ == "__main__":
    try:
        # 可以选择运行所有测试或特定测试
        # run_all_tests()
        
        # 或者只运行合同记忆测试
        test_contract_memory_flow()
    except Exception as e:
        logger.error(f"测试过程中发生错误: {e}")