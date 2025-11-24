import os
import sys
import logging
from langgraph.graph import StateGraph
from langchain_core.prompts import ChatPromptTemplate

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入LLMMap类和Agent类以及create_agent_node函数
from LLM_base.map import LLMMap
from LLM_base.Agent import Agent, create_agent_node

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_llm_map():
    """
    测试LLMMap类的所有功能，包括添加Agent节点
    """
    logger.info("开始测试LLMMap类")
    
    # 测试1: 初始化测试
    logger.info("=== 测试1: 初始化 ===")
    try:
        # 使用默认配置路径初始化
        llm_map = LLMMap()
        logger.info("初始化成功，配置数据类型: %s", type(llm_map.config))
        logger.info("map初始状态: %s", llm_map.map)
    except Exception as e:
        logger.error("初始化测试失败: %s", e)
        return False
    
    # 测试2: 检查空map
    logger.info("\n=== 测试2: 检查空map ===")
    try:
        result = llm_map.check_map()
        logger.info("检查空map结果: %s", result)
    except Exception as e:
        logger.error("检查空map测试失败: %s", e)
        return False
    
    # 测试3: 创建空map
    logger.info("\n=== 测试3: 创建空map ===")
    try:
        # 不再需要显式指定state_schema，map.py中会使用dict作为默认类型
        success = llm_map.set_map()
        logger.info("创建空map结果: %s", success)
        if success:
            logger.info("创建后的map类型: %s", type(llm_map.map))
    except Exception as e:
        logger.error("创建空map测试失败: %s", e)
        return False
    
    # 测试4: 检查创建的空map
    logger.info("\n=== 测试4: 检查创建的空map ===")
    try:
        result = llm_map.check_map()
        logger.info("检查创建的空map结果: %s", result)
    except Exception as e:
        logger.error("检查创建的空map测试失败: %s", e)
        return False
    
    # 测试5: 创建Agent节点
    logger.info("\n=== 测试5: 创建Agent节点 ===")
    try:
        # 创建Agent节点
        agent_node_func = create_agent_node()
        logger.info("创建Agent节点结果: %s", "成功" if agent_node_func is not None else "失败")
    except Exception as e:
        logger.error("创建Agent节点测试失败: %s", e)
    
    # 测试6: 添加普通节点到map
    logger.info("\n=== 测试6: 添加普通节点到map ===")
    try:
        # 定义一些简单的函数作为节点
        def start_node(state):
            logger.info("执行start_node")
            return {"step": "start", "input": state.get("input", "")}
        
        def end_node(state):
            logger.info("执行end_node")
            return {"step": "end", "result": state.get("response", state.get("input", ""))}
        
        # 添加节点到map
        success_start = llm_map.add_node("start", start_node)
        success_end = llm_map.add_node("end", end_node)
        logger.info("添加普通节点结果 - start: %s, end: %s", success_start, success_end)
    except Exception as e:
        logger.error("添加普通节点测试失败: %s", e)
        return False
    
    # 测试7: 添加Agent节点到map
    logger.info("\n=== 测试7: 添加Agent节点到map ===")
    try:
        if agent_node_func is not None:
            # 将Agent节点添加到map
            success_agent = llm_map.add_node("agent", agent_node_func)
            logger.info("添加Agent节点结果: %s", success_agent)
            
            # 添加边连接节点
            if success_agent and llm_map.map is not None:
                llm_map.map.add_edge("start", "agent")
                llm_map.map.add_edge("agent", "end")
                logger.info("添加边成功: start->agent->end")
                
                # 设置入口点
                llm_map.map.set_entry_point("start")
                logger.info("设置入口点为'start'")
        else:
            logger.warning("跳过添加Agent节点: Agent节点未成功创建")
    except Exception as e:
        logger.error("添加Agent节点测试失败: %s", e)
    
    # 测试8: 检查包含Agent节点的map结构
    logger.info("\n=== 测试8: 检查包含Agent节点的map结构 ===")
    try:
        result = llm_map.check_map()
        logger.info("检查map结构结果: %s", result)
    except Exception as e:
        logger.error("检查map结构测试失败: %s", e)
        return False
    
    # 测试9: 执行包含Agent节点的map进行简单输入测试
    logger.info("\n=== 测试9: 执行包含Agent节点的map进行简单输入测试 ===")
    try:
        if llm_map.map is not None:
            # 编译图
            try:
                logger.info("尝试编译图...")
                app = llm_map.map.compile()  # 只有编译后的app才有invoke方法
                logger.info("编译图成功")
                
                # 简单输入测试
                test_input = {"input": "你好！", "prompt": "请用一句话打招呼"}
                logger.info("执行简单输入测试，输入: %s", test_input)
                
                # 执行编译后的图 (app才有invoke方法)
                if hasattr(app, 'invoke'):
                    result = app.invoke(test_input)
                    logger.info("图执行结果: %s", result)
                    
                    # 检查是否有预期的输出
                    if "response" in result or "output" in result:
                        output_key = "response" if "response" in result else "output"
                        logger.info("测试成功! 收到回复: %s", result[output_key][:100])
                    else:
                        logger.info("未找到标准响应键，但执行成功")
                else:
                    logger.warning("编译后的图对象没有invoke方法")
            except Exception as e:
                logger.warning("图编译或执行失败: %s", str(e))
        else:
            logger.info("跳过测试：map对象不存在")
    except Exception as e:
        logger.error("执行测试9失败: %s", str(e))
    
    logger.info("\nLLMMap测试完成")
    return True

def test_agent():
    """
    测试Agent类的所有功能
    """
    logger.info("开始测试Agent类")
    
    # 测试1: 初始化测试
    logger.info("=== 测试1: 初始化 ===")
    try:
        # 使用默认配置路径初始化
        agent = Agent()
        logger.info("初始化成功，配置数据类型: %s", type(agent.config))
        logger.info("LLM初始状态: %s", agent.llm)
        logger.info("API配置: API_KEY=%s..., API_URL=%s", agent.api_key[:10], agent.api_url)
    except Exception as e:
        logger.error("初始化测试失败: %s", e)
        return False
    
    # 测试2: 尝试在创建LLM前生成回复（应该失败）
    logger.info("\n=== 测试2: LLM未初始化时生成回复 ===")
    try:
        response = agent.generate_response("你好，请介绍一下自己")
        if response is None:
            logger.info("预期行为: 未创建LLM时无法生成回复")
        else:
            logger.warning("意外行为: 未创建LLM时生成了回复")
    except Exception as e:
        logger.error("测试失败: %s", e)
        return False
    
    # 测试3: 创建LLM
    logger.info("\n=== 测试3: 创建LLM ===")
    try:
        # 尝试创建LLM实例
        success = agent.create_llm()
        logger.info("创建LLM结果: %s", success)
        if success:
            logger.info("LLM类型: %s", type(agent.llm))
        else:
            logger.warning("LLM创建失败，可能缺少必要的依赖包")
    except Exception as e:
        logger.error("创建LLM测试失败: %s", e)
        # 即使创建LLM失败，我们仍然继续测试其他功能
    
    # 测试4: 生成回复（如果LLM已成功创建）
    logger.info("\n=== 测试4: 生成回复 ===")
    try:
        if agent.llm is not None:
            # 发送一个简单的提示词
            prompt = "你好，请简要介绍一下你自己"
            logger.info("发送提示词: %s", prompt)
            response = agent.generate_response(prompt)
            if response:
                logger.info("成功获取回复，回复长度: %d 字符", len(response))
                logger.info("回复内容预览: %s...", response[:100])
            else:
                logger.warning("未能获取回复")
        else:
            logger.info("跳过测试：LLM未成功创建")
    except Exception as e:
        logger.error("生成回复测试失败: %s", e)
    
    # 测试5: 测试不同的提示词类型
    logger.info("\n=== 测试5: 不同提示词类型 ===")
    try:
        if agent.llm is not None:
            # 测试一个简单的问题
            prompt = "1+1等于多少？"
            logger.info("发送数学问题: %s", prompt)
            response = agent.generate_response(prompt)
            if response:
                logger.info("数学问题回复: %s", response[:100])
        else:
            logger.info("跳过测试：LLM未成功创建")
    except Exception as e:
        logger.error("不同提示词类型测试失败: %s", e)
    
    # 测试6: 测试错误处理 - 空提示词
    logger.info("\n=== 测试6: 错误处理 - 空提示词 ===")
    try:
        if agent.llm is not None:
            response = agent.generate_response("")
            logger.info("空提示词处理结果: %s", "成功" if response is not None else "失败")
        else:
            logger.info("跳过测试：LLM未成功创建")
    except Exception as e:
        logger.error("空提示词测试失败: %s", e)
    
    # 测试7: 创建Agent节点
    logger.info("\n=== 测试7: 创建Agent节点 ===")
    try:
        # 尝试创建Agent节点
        agent_node = create_agent_node()
        logger.info("创建Agent节点结果: %s", "成功" if agent_node is not None else "失败")
        if agent_node:
            logger.info("Agent节点类型: %s", type(agent_node))
        else:
            logger.warning("Agent节点创建失败，可能缺少必要的依赖包")
    except Exception as e:
        logger.error("创建Agent节点测试失败: %s", e)
    
    # 测试8: 测试Agent节点处理 - 使用prompt键
    logger.info("\n=== 测试8: 测试Agent节点处理(prompt键) ===")
    try:
        if agent_node is not None:
            # 测试基本输入处理
            test_state = {"prompt": "你好，请简要介绍一下你自己"}
            logger.info("发送状态: %s", test_state)
            result = agent_node(test_state)
            if result:
                logger.info("节点处理结果: 状态=%s", result.get("status"))
                if result.get("response"):
                    logger.info("节点生成回复预览: %s...", result["response"][:100])
            else:
                logger.warning("节点处理失败，无返回结果")
        else:
            logger.info("跳过测试：Agent节点未成功创建")
    except Exception as e:
        logger.error("Agent节点处理测试失败: %s", e)
    
    # 测试9: 测试Agent节点不同输入键和错误处理
    logger.info("\n=== 测试9: 测试Agent节点不同输入键和错误处理 ===")
    try:
        if agent_node is not None:
            # 测试使用input键
            test_state_input = {"input": "1+1等于多少？"}
            logger.info("使用input键发送: %s", test_state_input)
            result = agent_node(test_state_input)
            logger.info("使用input键处理结果: 状态=%s", result.get("status"))
            
            # 测试使用query键
            test_state_query = {"query": "什么是人工智能？"}
            logger.info("使用query键发送: %s", test_state_query)
            result = agent_node(test_state_query)
            logger.info("使用query键处理结果: 状态=%s", result.get("status"))
            
            # 测试无有效输入键
            test_state_invalid = {"invalid_key": "这是无效键"}
            logger.info("使用无效键发送: %s", test_state_invalid)
            result = agent_node(test_state_invalid)
            logger.info("无效键处理结果: 状态=%s, 错误=%s", 
                        result.get("status"), result.get("error"))
        else:
            logger.info("跳过测试：Agent节点未成功创建")
    except Exception as e:
        logger.error("Agent节点不同输入键测试失败: %s", e)
    
    logger.info("\nAgent测试完成")
    return True

if __name__ == "__main__":
    logger.info("运行测试套件")
    
    # 可以选择要运行的测试
    run_map_test = True  # 设置为False可以跳过LLMMap测试
    run_agent_test = False  # 设置为True可以运行Agent测试
    
    map_success = True
    agent_success = True
    
    if run_map_test:
        logger.info("\n===========================")
        logger.info("运行LLMMap测试")
        logger.info("===========================")
        map_success = test_llm_map()
    
    if run_agent_test:
        logger.info("\n===========================")
        logger.info("运行Agent测试")
        logger.info("===========================")
        agent_success = test_agent()
    
    logger.info("\n===========================")
    logger.info("测试结果汇总")
    logger.info("===========================")
    
    if run_map_test:
        logger.info("LLMMap测试: %s", "通过" if map_success else "失败")
    if run_agent_test:
        logger.info("Agent测试: %s", "通过" if agent_success else "失败")
    
    if (run_map_test and not map_success) or (run_agent_test and not agent_success):
        logger.error("部分测试失败")
    else:
        logger.info("所有测试通过")