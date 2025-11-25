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

# 添加RAG相关导入
try:
    from LLM_base.RAG import RAG, rag_node
    RAG_AVAILABLE = True
except ImportError:
    logging.warning("RAG模块未找到或导入失败")
    RAG_AVAILABLE = False

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_llm_map():
    """
    测试LLMMap类的所有功能，包括添加Agent节点和RAG节点
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
    
    # 测试5.1: 创建RAG节点
    logger.info("\n=== 测试5.1: 创建RAG节点 ===")
    try:
        # 初始化RAG实例
        rag_instance = None
        rag_node_func = None
        
        if RAG_AVAILABLE:
            rag_instance = RAG()
            # 获取RAG节点函数
            if hasattr(rag_instance, 'rag_node'):
                rag_node_func = lambda state: rag_instance.rag_node(state)
            elif 'rag_node' in globals() or 'rag_node' in locals():
                rag_node_func = lambda state: rag_node(state, rag_instance=rag_instance)
            else:
                # 尝试直接使用导入的rag_node函数
                import inspect
                if callable(rag_node):
                    sig = inspect.signature(rag_node)
                    if 'rag_instance' in sig.parameters:
                        rag_node_func = lambda state: rag_node(state, rag_instance=rag_instance)
                    else:
                        rag_node_func = rag_node
            
            logger.info("创建RAG节点结果: %s", "成功" if rag_node_func is not None else "失败")
        else:
            logger.warning("RAG模块不可用，跳过RAG节点创建")
    except Exception as e:
        logger.error("创建RAG节点测试失败: %s", e)
    
    # 测试6: 添加普通节点到map
    logger.info("\n=== 测试6: 添加普通节点到map ===")
    try:
        # 定义一些简单的函数作为节点
        def start_node(state):
            logger.info("执行start_node")
            return {"step": "start", "input": state.get("input", ""), "prompt": state.get("prompt", "")}
        
        def end_node(state):
            logger.info("执行end_node")
            # 如果有RAG检索的结果，将其加入输出
            if "retrieved_context" in state:
                logger.info(f"从RAG节点获取到检索结果，包含{len(state['retrieved_context'].get('results', []))}个文档")
            return {"step": "end", "result": state.get("response", state.get("input", "")), "rag_context": state.get("retrieved_context")}
        
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
        else:
            logger.warning("跳过添加Agent节点: Agent节点未成功创建")
    except Exception as e:
        logger.error("添加Agent节点测试失败: %s", e)
    
    # 测试7.1: 添加RAG节点到map
    logger.info("\n=== 测试7.1: 添加RAG节点到map ===")
    try:
        if rag_node_func is not None:
            # 将RAG节点添加到map
            success_rag = llm_map.add_node("rag", rag_node_func)
            logger.info("添加RAG节点结果: %s", success_rag)
        else:
            logger.warning("跳过添加RAG节点: RAG节点未成功创建")
    except Exception as e:
        logger.error("添加RAG节点测试失败: %s", e)
    
    # 测试8: 设置图的边和入口点
    logger.info("\n=== 测试8: 设置图的边和入口点 ===")
    try:
        if llm_map.map is not None:
            # 根据可用节点设置不同的边连接策略
            if rag_node_func is not None and agent_node_func is not None:
                # RAG + Agent 组合流程
                llm_map.map.add_edge("start", "rag")
                llm_map.map.add_edge("rag", "agent")
                llm_map.map.add_edge("agent", "end")
                logger.info("添加边成功: start->rag->agent->end")
            elif rag_node_func is not None:
                # 仅RAG流程
                llm_map.map.add_edge("start", "rag")
                llm_map.map.add_edge("rag", "end")
                logger.info("添加边成功: start->rag->end")
            elif agent_node_func is not None:
                # 仅Agent流程
                llm_map.map.add_edge("start", "agent")
                llm_map.map.add_edge("agent", "end")
                logger.info("添加边成功: start->agent->end")
            else:
                # 基本流程
                llm_map.map.add_edge("start", "end")
                logger.info("添加边成功: start->end")
            
            # 设置入口点
            llm_map.map.set_entry_point("start")
            logger.info("设置入口点为'start'")
        else:
            logger.warning("map对象不存在，跳过设置边")
    except Exception as e:
        logger.error("设置边测试失败: %s", e)
    
    # 测试9: 检查map结构
    logger.info("\n=== 测试9: 检查map结构 ===")
    try:
        result = llm_map.check_map()
        logger.info("检查map结构结果: %s", result)
    except Exception as e:
        logger.error("检查map结构测试失败: %s", e)
        return False
    
    # 测试10: 执行包含RAG和Agent节点的map进行测试
    logger.info("\n=== 测试10: 执行map进行测试 ===")
    try:
        if llm_map.map is not None:
            # 编译图
            try:
                logger.info("尝试编译图...")
                app = llm_map.map.compile()  # 只有编译后的app才有invoke方法
                logger.info("编译图成功")
                
                # 准备测试输入，包含query用于RAG检索
                test_input = {
                    "input": "人工智能产业园的相关信息", 
                    "prompt": "请基于提供的上下文信息，介绍人工智能产业园",
                    "query": "人工智能产业园"
                }
                logger.info("执行测试，输入: %s", test_input)
                
                # 执行编译后的图
                if hasattr(app, 'invoke'):
                    result = app.invoke(test_input)
                    logger.info("图执行结果: %s", "成功" if result else "失败")
                    
                    # 检查结果中的关键信息
                    if result:
                        if "retrieved_context" in result:
                            rag_results = result["retrieved_context"]
                            logger.info("RAG检索结果: %s个相关文档", len(rag_results.get("results", [])))
                            if rag_results.get("results"):
                                logger.info("第一个检索文档预览: %s...", rag_results["results"][0]["content"][:100])
                        
                        if "response" in result:
                            logger.info("最终响应预览: %s...", result["response"][:100])
                        
                        logger.info("测试成功完成")
                else:
                    logger.warning("编译后的图对象没有invoke方法")
            except Exception as e:
                logger.warning("图编译或执行失败: %s", str(e))
        else:
            logger.info("跳过测试：map对象不存在")
    except Exception as e:
        logger.error("执行测试10失败: %s", str(e))
    
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

def test_rag():
    """
    测试RAG类的所有功能，包括新添加的节点和工具包装功能
    """
    logger.info("开始测试RAG类")
    
    # 尝试导入RAG类和新添加的函数
    try:
        # 添加当前目录到Python路径（确保能正确导入RAG模块）
        if __name__ == "__main__":
            sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        
        from LLM_base.RAG import RAG, rag_node, create_rag_tool, LANGGRAPH_AVAILABLE
        logger.info("成功导入RAG类和相关函数")
    except ImportError as e:
        logger.error(f"导入RAG类失败: {e}")
        return False
    
    # 测试1: 初始化测试
    logger.info("\n=== 测试1: 初始化 ===")
    try:
        # 使用默认配置路径初始化
        rag = RAG()
        logger.info("初始化成功")
        # 检查基本属性是否存在
        required_attrs = ['vectorstore_info', 'vectorstore']
        for attr in required_attrs:
            if hasattr(rag, attr):
                logger.info(f"属性 {attr} 存在")
            else:
                logger.warning(f"属性 {attr} 不存在")
    except Exception as e:
        logger.error(f"初始化测试失败: {e}")
        return False
    
    # 测试2: 检查向量库是否存在
    logger.info("\n=== 测试2: 检查向量库是否存在 ===")
    try:
        # 调用检查向量库是否存在的方法
        if hasattr(rag, 'check_vectorstore_exists'):
            exists = rag.check_vectorstore_exists()
            logger.info(f"向量库存在状态: {exists}")
        else:
            logger.warning("check_vectorstore_exists方法不存在")
    except Exception as e:
        logger.error(f"检查向量库测试失败: {e}")
    
    # 测试3: 创建向量库
    logger.info("\n=== 测试3: 创建向量库 ===")
    try:
        # 调用创建向量库的方法
        if hasattr(rag, 'create_vectorstore'):
            created = rag.create_vectorstore()
            logger.info(f"创建向量库结果: {created}")
        else:
            logger.warning("create_vectorstore方法不存在")
    except Exception as e:
        logger.error(f"创建向量库测试失败: {e}")
    
    # 测试4: 获取向量库句柄
    logger.info("\n=== 测试4: 获取向量库句柄 ===")
    try:
        # 调用获取向量库句柄的方法
        if hasattr(rag, 'get_vectorstore'):
            vectorstore = rag.get_vectorstore()
            logger.info(f"获取向量库句柄结果: {'成功' if vectorstore is not None else '失败'}")
        else:
            logger.warning("get_vectorstore方法不存在")
    except Exception as e:
        logger.error(f"获取向量库句柄测试失败: {e}")
    
    # 测试5: 添加文档到向量库
    logger.info("\n=== 测试5: 添加文档到向量库 ===")
    try:
        # 调用添加文档的方法
        if hasattr(rag, 'add_document_to_vectorstore'):
            # 尝试添加RAG/documents目录下的文档
            doc_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'LLM_base', 'RAG', 'documents', '投资合作协议2.docx')
            if os.path.exists(doc_path):
                added = rag.add_document_to_vectorstore(doc_path)
                logger.info(f"添加文档结果: {added}")
            else:
                logger.warning(f"测试文档不存在: {doc_path}")
        else:
            logger.warning("add_document_to_vectorstore方法不存在")
    except Exception as e:
        logger.error(f"添加文档测试失败: {e}")
    
    # 测试6: 检索知识库
    logger.info("\n=== 测试6: 检索知识库 ===")
    try:
        # 调用检索知识库的方法
        if hasattr(rag, 'search_knowledge_base'):
            # 执行简单的查询
            query = "人工智能产业园"
            results = rag.search_knowledge_base(query, k=3)
            if results:
                logger.info(f"检索结果数量: {len(results.get('results', []))}")
                if results.get('results'):
                    for i, result in enumerate(results['results'][:2]):
                        logger.info(f"结果 {i+1} 内容预览: {result['content'][:100]}...")
                        logger.info(f"结果 {i+1} 相似度: {result['score']}")
                if results.get('sources'):
                    logger.info(f"来源文件数量: {len(results['sources'])}")
                    logger.info(f"来源文件列表: {results['sources']}")
            else:
                logger.warning("检索未返回结果")
        else:
            logger.warning("search_knowledge_base方法不存在")
    except Exception as e:
        logger.error(f"检索知识库测试失败: {e}")
    
    # 测试7: 错误处理 - 无效文件路径
    logger.info("\n=== 测试7: 错误处理 - 无效文件路径 ===")
    try:
        # 测试添加不存在的文件
        if hasattr(rag, 'add_document_to_vectorstore'):
            invalid_path = "不存在的文件.txt"
            added = rag.add_document_to_vectorstore(invalid_path)
            logger.info(f"添加无效文件结果: {added}")
    except Exception as e:
        logger.error(f"错误处理测试失败: {e}")
    
    # 测试8: RAG包装成langgraph节点
    logger.info("\n=== 测试8: RAG包装成langgraph节点 ===")
    try:
        #print(globals())
        if LANGGRAPH_AVAILABLE:

            logger.info("langgraph可用，开始测试rag_node函数")
            
            # 准备测试状态
            test_state = {'query': "人工智能产业园"}
            
            # 测试rag_node函数
            result_state = rag_node(test_state, rag_instance=rag, k=2)
            
            if 'retrieved_context' in result_state and result_state['retrieved_context'] is not None:
                logger.info("RAG节点测试成功: 成功获取检索结果")
                results = result_state['retrieved_context']
                if results.get('results'):
                    logger.info(f"RAG节点检索到{len(results['results'])}个相关文档")
                    logger.info(f"第二个结果内容预览: {results['results'][1]['content'][:100]}...")
            else:
                logger.warning("RAG节点测试未获取到检索结果")
        else:
            logger.warning("langgraph不可用或rag_node函数未导入，跳过节点测试")
    except Exception as e:
        logger.error(f"RAG节点测试失败: {e}")
    
    # 测试9: RAG包装成agent工具
    logger.info("\n=== 测试9: RAG包装成agent工具 ===")
    try:
        if LANGGRAPH_AVAILABLE:
            logger.info("langchain工具可用，开始测试create_rag_tool函数")
            
            # 测试create_rag_tool函数
            rag_tool = create_rag_tool(rag_instance=rag)
            
            if rag_tool is not None:
                logger.info(f"成功创建RAG工具: {rag_tool.name}")
                logger.info(f"工具描述: {rag_tool.description}")
                
                # 测试工具调用
                try:
                    tool_result = rag_tool.run({
                        'query': "人工智能产业园",
                        'k': 2
                    })
                    logger.info(f"工具调用成功，结果长度: {len(str(tool_result))} 字符")
                    logger.info(f"工具结果预览: {str(tool_result)[:200]}...")
                except Exception as e:
                    logger.error(f"工具调用失败: {e}")
            else:
                logger.warning("创建RAG工具失败")
        else:
            logger.warning("langchain工具不可用或create_rag_tool函数未导入，跳過工具测试")
    except Exception as e:
        logger.error(f"RAG工具测试失败: {e}")
    
    logger.info("\nRAG测试完成")
    return True

if __name__ == "__main__":
    logger.info("运行测试套件")
    
    # 可以选择要运行的测试
    run_map_test = True  # 设置为False可以跳过LLMMap测试
    run_agent_test = False  # 设置为True可以运行Agent测试
    run_rag_test = False  # 设置为True可以运行RAG测试
    
    map_success = True
    agent_success = True
    rag_success = True
    
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
        logger.info("\n===========================")
        logger.info("运行LLMMap测试")
        logger.info("===========================")
        map_success = test_llm_map()
    
    if run_agent_test:
        logger.info("\n===========================")
        logger.info("运行Agent测试")
        logger.info("===========================")
        agent_success = test_agent()
    
    if run_rag_test:
        logger.info("\n===========================")
        logger.info("运行RAG测试")
        logger.info("===========================")
        rag_success = test_rag()
    
    logger.info("\n===========================")
    logger.info("测试结果汇总")
    logger.info("===========================")
    
    if run_map_test:
        logger.info("LLMMap测试: %s", "通过" if map_success else "失败")
    if run_agent_test:
        logger.info("Agent测试: %s", "通过" if agent_success else "失败")
    if run_rag_test:
        logger.info("RAG测试: %s", "通过" if rag_success else "失败")
    
    all_tests_passed = True
    if run_map_test and not map_success:
        all_tests_passed = False
    if run_agent_test and not agent_success:
        all_tests_passed = False
    if run_rag_test and not rag_success:
        all_tests_passed = False
    
    if not all_tests_passed:
        logger.error("部分测试失败")
    else:
        logger.info("所有测试通过")