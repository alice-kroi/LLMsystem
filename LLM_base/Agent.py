import os
import logging
from typing import Optional, Any, Callable, Dict
import sys

# 导入之前创建的配置加载模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tool.config_load import load_config_to_env

# 导入langgraph相关库（假设已经安装）
try:
    from langgraph.graph import StateGraph
    LANGGRAPH_AVAILABLE = True
except ImportError:
    logging.warning("langgraph库未安装，某些功能可能无法使用")
    LANGGRAPH_AVAILABLE = False

# 导入langchain相关库用于创建LLM
try:
    from langchain_core.language_models import BaseLanguageModel
    from langchain_community.chat_models import ChatZhipuAI  # 智谱AI的LLM
    LANGCHAIN_AVAILABLE = True
except ImportError:
    logging.warning("langchain库未安装，LLM功能可能无法使用")
    LANGCHAIN_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Agent:
    def __init__(self, config_path=None):
        """
        初始化Agent类
        
        Args:
            config_path (str, optional): 配置文件路径，默认为None
        """
        # 利用config_load函数加载配置
        self.config = load_config_to_env(config_path=config_path, return_dict=True)
        if self.config:
            logger.info("配置文件加载成功")
            # 从配置中提取API_KEY和API_URL
            self.api_key = self.config.get('API_KEY', '505518dcc98644f3a1db58caad756250.gqmz7KqgySG2UguA')
            self.api_url = self.config.get('API_URL', 'https://open.bigmodel.cn/api/paas/v4/')
        else:
            logger.warning("配置文件加载失败或为空，使用默认API配置")
            # 使用默认值
            self.api_key = '505518dcc98644f3a1db58caad756250.gqmz7KqgySG2UguA'
            self.api_url = 'https://open.bigmodel.cn/api/paas/v4/'
        
        # 创建空的LLM属性
        self.llm: Optional[BaseLanguageModel] = None
        logger.info("初始化空LLM属性完成")
    
    def create_llm(self, model_name: str = "glm-4", **kwargs) -> bool:
        """
        创建LLM实例并保存到llm属性
        
        Args:
            model_name (str): 模型名称，默认为"glm-4"
            **kwargs: 其他传递给LLM的参数
            
        Returns:
            bool: 创建是否成功
        """
        try:
            if not LANGCHAIN_AVAILABLE:
                logger.error("langchain库未安装，无法创建LLM")
                return False
            
            # 创建智谱AI的LLM实例
            self.llm = ChatZhipuAI(
                api_key=self.api_key,
                base_url=self.api_url,
                model=model_name,
                **kwargs
            )
            
            logger.info(f"成功创建LLM实例，模型名称: {model_name}")
            return True
        except Exception as e:
            logger.error(f"创建LLM实例时出错: {e}")
            return False
    
    def generate_response(self, prompt: str) -> Optional[str]:
        """
        输入提示词，获取LLM回复
        
        Args:
            prompt (str): 提示词
            
        Returns:
            Optional[str]: LLM的回复，如果失败则返回None
        """
        # 检查LLM属性是否存在
        if self.llm is None:
            logger.error("LLM实例不存在，请先调用create_llm方法")
            return None
        
        try:
            # 将提示词输入LLM并获取回复
            response = self.llm.invoke(prompt)
            
            # 提取回复内容（根据不同LLM返回格式可能需要调整）
            if hasattr(response, 'content'):
                reply = response.content
            elif isinstance(response, dict) and 'content' in response:
                reply = response['content']
            else:
                reply = str(response)
            
            logger.info("成功获取LLM回复")
            return reply
        except Exception as e:
            logger.error(f"获取LLM回复时出错: {e}")
            return None
        

def create_agent_node(config_path: Optional[str] = None, model_name: str = "glm-4", **kwargs) -> Optional[Callable[[Dict[str, Any]], Dict[str, Any]]]:
    """
    将Agent类初始化创建LLM后，包装成langgraph的一个图节点并返回
    
    Args:
        config_path (str, optional): 配置文件路径，默认为None
        model_name (str): 模型名称，默认为"glm-4"
        **kwargs: 传递给LLM初始化的其他参数
        
    Returns:
        Optional[Callable[[Dict[str, Any]], Dict[str, Any]]]: langgraph图节点函数，如果创建失败则返回None
    """
    if not LANGGRAPH_AVAILABLE:
        logger.error("langgraph库未安装，无法创建图节点")
        return None
    
    try:
        # 初始化Agent实例
        agent = Agent(config_path=config_path)
        
        # 创建LLM
        if not agent.create_llm(model_name=model_name, **kwargs):
            logger.error("LLM创建失败，无法创建图节点")
            return None
        
        logger.info("Agent初始化成功，开始创建langgraph图节点")
        
        # 定义langgraph节点函数
        def agent_node(state: Dict[str, Any]) -> Dict[str, Any]:
            """
            langgraph图节点函数，处理状态并生成响应
            
            Args:
                state (Dict[str, Any]): 包含输入信息的状态字典
                
            Returns:
                Dict[str, Any]: 更新后的状态字典
            """
            try:
                # 从状态中提取输入提示
                # 尝试多种可能的输入键名
                prompt = state.get('prompt')
                if prompt is None:
                    prompt = state.get('input')
                if prompt is None:
                    prompt = state.get('query')
                if prompt is None:
                    logger.error("在状态字典中找不到有效的提示词")
                    return {**state, 'error': "找不到有效的提示词"}
                
                # 使用Agent生成回复
                response = agent.generate_response(prompt)
                
                if response is not None:
                    # 更新状态字典
                    updated_state = {
                        **state,
                        'response': response,
                        'agent_output': response,
                        'status': 'success'
                    }
                    logger.info("Agent节点成功处理输入并生成回复")
                    return updated_state
                else:
                    logger.error("Agent生成回复失败")
                    return {**state, 'error': "Agent生成回复失败", 'status': 'error'}
            except Exception as e:
                logger.error(f"Agent节点执行时出错: {e}")
                return {**state, 'error': str(e), 'status': 'error'}
        
        logger.info("langgraph图节点创建成功")
        return agent_node
    except Exception as e:
        logger.error(f"创建Agent节点时出错: {e}")
        return None