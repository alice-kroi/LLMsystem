import os
import json
import uuid
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
    # 添加通用LLM导入
    from langchain_openai import ChatOpenAI  # 通用OpenAI兼容接口
    # 导入memory相关模块
    from langchain_classic.memory import ConversationBufferMemory
    LANGCHAIN_AVAILABLE = True
except ImportError:
    logging.warning("langchain库未安装，LLM功能可能无法使用")
    LANGCHAIN_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Agent:
    def __init__(self, config_path="E:\GitHub\config.yaml", model_type="glm"):
        """
        初始化Agent类
        
        Args:
            config_path (str, optional): 配置文件路径，默认为"E:\GitHub\config.yaml"
            model_type (str, optional): 模型类型，默认为"glm"，可选值为"glm"或"doubao"
        """
        # 利用config_load函数加载配置
        self.config = load_config_to_env(config_path=config_path, return_dict=True)
        
        # 根据model_type选择不同的API配置
        if model_type == "doubao":
            if self.config:
                logger.info("配置文件加载成功，使用豆包API配置")
                # 从配置中提取豆包的API_KEY和API_URL
                self.api_key = self.config.get('Doubao_API_KEY', 'cf1d0e35-d99b-4189-8c69-92a175619833')
                self.api_url = self.config.get('Doubao_API_URL', 'https://ark.cn-beijing.volces.com/api/v3')
            else:
                logger.warning("配置文件加载失败或为空，使用默认豆包API配置")
                # 使用豆包默认值
                self.api_key = 'cf1d0e35-d99b-4189-8c69-92a175619833'
                self.api_url = 'https://ark.cn-beijing.volces.com/api/v3'
        else:  # glm或其他情况，使用智谱AI的配置
            if self.config:
                logger.info("配置文件加载成功，使用智谱AI API配置")
                # 从配置中提取API_KEY和API_URL
                self.api_key = self.config.get('API_KEY', '505518dcc98644f3a1db58caad756250.gqmz7KqgySG2UguA')
                self.api_url = self.config.get('API_URL', 'https://open.bigmodel.cn/api/paas/v4/')
            else:
                logger.warning("配置文件加载失败或为空，使用默认智谱AI API配置")
                # 使用默认值
                self.api_key = '505518dcc98644f3a1db58caad756250.gqmz7KqgySG2UguA'
                self.api_url = 'https://open.bigmodel.cn/api/paas/v4/'
        

        # 保存model_type供后续使用
        self.model_type = model_type
        logger.info(f"初始化完成，当前模型类型: {model_type}")
        
        # 创建空的LLM属性
        self.llm: Optional[BaseLanguageModel] = None
        logger.info("初始化空LLM属性完成")
        
        # 初始化记忆管理相关属性
        self.memory_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'memory')
        # 确保memory目录存在
        if not os.path.exists(self.memory_dir):
            os.makedirs(self.memory_dir)
            logger.info(f"创建memory目录: {self.memory_dir}")
        
        # 用于存储不同conversation_id对应的memory实例的字典
        self.memories: Dict[str, ConversationBufferMemory] = {}
    
    def create_llm(self, model_name: str = "glm-4", provider: str = "zhipu", **kwargs) -> bool:
        """
        创建LLM实例并保存到llm属性
        
        Args:
            model_name (str): 模型名称，默认为"glm-4"
            provider (str): LLM提供商，可选值: "zhipu"(智谱AI) 或 "openai"(通用OpenAI兼容接口)，默认为"zhipu"
            **kwargs: 其他传递给LLM的参数
            
        Returns:
            bool: 创建是否成功
        """
        try:
            if not LANGCHAIN_AVAILABLE:
                logger.error("langchain库未安装，无法创建LLM")
                return False
            
            provider = provider.lower()
            
            if provider == "zhipu":
                # 创建智谱AI的LLM实例
                self.llm = ChatZhipuAI(
                    api_key=self.api_key,
                    base_url=self.api_url,
                    model=model_name,
                    **kwargs
                )
                logger.info(f"成功创建智谱AI LLM实例，模型名称: {model_name}")
            
            elif provider == "doubao":
                # 创建通用OpenAI兼容接口的LLM实例
                # 从kwargs或配置中获取API密钥和基础URL
                api_key = self.api_key,
                base_url = self.api_url,
                
                # 基本参数
                llm_params = {
                    'model': model_name,
                    'openai_api_key': self.api_key,
                    'openai_api_base': self.api_url,
                }
                
                # 添加可选参数
                if api_key:
                    llm_params['api_key'] = api_key
                if base_url:
                    llm_params['base_url'] = base_url
                
                # 添加其他传入的参数
                llm_params.update(kwargs)
                
                self.llm = ChatOpenAI(**llm_params)
                logger.info(f"成功创建通用OpenAI兼容LLM实例，模型名称: {model_name}")
            
            else:
                logger.error(f"不支持的LLM提供商: {provider}，请使用 'zhipu' 或 'openai'")
                return False
            
            return True
        except Exception as e:
            logger.error(f"创建LLM实例时出错: {e}")
            return False
    
    def _get_memory_file_path(self, conversation_id: str) -> str:
        """
        获取记忆文件的路径
        
        Args:
            conversation_id (str): 对话ID
            
        Returns:
            str: 记忆文件的完整路径
        """
        return os.path.join(self.memory_dir, f"{conversation_id}.json")
    
    def _load_memory_from_file(self, conversation_id: str) -> ConversationBufferMemory:
        """
        从文件加载记忆
        
        Args:
            conversation_id (str): 对话ID
            
        Returns:
            ConversationBufferMemory: 加载后的记忆实例
        """
        file_path = self._get_memory_file_path(conversation_id)
        memory = ConversationBufferMemory()
        
        # 如果文件存在，加载内容
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # 加载对话历史
                    if 'conversations' in data:
                        for conv in data['conversations']:
                            if 'Human' in conv and 'AI' in conv:
                                memory.save_context(
                                    {'input': conv['Human']},
                                    {'output': conv['AI']}
                                )
                logger.info(f"成功从文件加载记忆: {file_path}")
            except Exception as e:
                logger.error(f"从文件加载记忆时出错: {e}")
        else:
            # 文件不存在，创建空文件
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump({'conversations': []}, f, ensure_ascii=False, indent=2)
                logger.info(f"创建新的空记忆文件: {file_path}")
            except Exception as e:
                logger.error(f"创建空记忆文件时出错: {e}")
        
        return memory
    
    def _save_memory_to_file(self, conversation_id: str, memory: ConversationBufferMemory):
        """
        保存记忆到文件
        
        Args:
            conversation_id (str): 对话ID
            memory (ConversationBufferMemory): 记忆实例
        """
        file_path = self._get_memory_file_path(conversation_id)
        try:
            # 直接从memory获取对话历史，而不是解析格式
            memory_variables = memory.load_memory_variables({})
            
            # 创建新的对话列表
            conversations = []
            
            # 使用memory的chat_memory.messages直接获取消息
            if hasattr(memory, 'chat_memory') and hasattr(memory.chat_memory, 'messages'):
                messages = memory.chat_memory.messages
                # 处理消息对（Human-AI）
                i = 0
                while i < len(messages):
                    if i + 1 < len(messages):
                        # 检查是否为Human-AI对
                        if messages[i].type == 'human' and messages[i+1].type == 'ai':
                            conversations.append({
                                'Human': messages[i].content,
                                'AI': messages[i+1].content
                            })
                            i += 2  # 跳过这对消息
                        else:
                            i += 1  # 不匹配则前进一位
                    else:
                        break
                logger.info(f"从chat_memory中提取了{len(conversations)}条对话记录")
            else:
                # 备用方法：使用历史文本解析
                if 'history' in memory_variables and memory_variables['history']:
                    history = memory_variables['history']
                    # 更健壮的解析方式
                    lines = history.split('\n')
                    current_conv = {}
                    
                    for line in lines:
                        line = line.strip()
                        if not line:
                            continue
                            
                        if line.startswith('Human:'):
                            # 如果已经有Human但没有匹配的AI，说明之前的AI回复被截断
                            if 'Human' in current_conv:
                                # 将不完整的对话记录下来，然后开始新的
                                conversations.append(current_conv.copy())
                            current_conv = {'Human': line[7:].strip()}
                        elif line.startswith('AI:') and 'Human' in current_conv:
                            # 保存完整的对话对
                            current_conv['AI'] = line[3:].strip()
                            conversations.append(current_conv.copy())
                            current_conv = {}
                    
                    # 记录最后一个可能不完整的对话
                    if current_conv:
                        conversations.append(current_conv)
                logger.info(f"从history文本中解析了{len(conversations)}条对话记录")
            
            # 保存到文件，确保完整写入
            temp_file = file_path + '.tmp'
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump({'conversations': conversations}, f, ensure_ascii=False, indent=2)
                # 强制刷新缓冲区
                f.flush()
                os.fsync(f.fileno())
            
            # 原子性替换文件，确保文件完整性
            os.replace(temp_file, file_path)
            logger.info(f"成功保存记忆到文件: {file_path}，共{len(conversations)}条对话")
            
        except Exception as e:
            logger.error(f"保存记忆到文件时出错: {e}")
            # 清理临时文件
            if os.path.exists(file_path + '.tmp'):
                try:
                    os.remove(file_path + '.tmp')
                except:
                    pass
    
    def generate_response(self, prompt: str, conversation_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        输入提示词，获取LLM回复，并管理对话记忆
        
        Args:
            prompt (str): 提示词
            conversation_id (str, optional): 对话ID，如果为None则生成新的ID
            
        Returns:
            Optional[Dict[str, Any]]: 包含回复内容和conversation_id的字典，如果失败则返回None
        """
        # 检查LLM是否存在
        if self.llm is None:
            # LLM不存在时
            if conversation_id is None:
                # conversation_id不存在时，创建LLM，生成新的conversation_id，创建空文件
                if not self.create_llm():
                    logger.error("LLM创建失败")
                    return None
                # 生成新的conversation_id
                conversation_id = str(uuid.uuid4())
                logger.info(f"生成新的对话ID: {conversation_id}")
                # 为新ID创建空文件
                memory = ConversationBufferMemory()
                self.memories[conversation_id] = memory
                # 保存空记忆到文件
                self._save_memory_to_file(conversation_id, memory)
            else:
                # conversation_id存在时，创建LLM，加载对应记忆文件
                if not self.create_llm():
                    logger.error("LLM创建失败")
                    return None
                # 加载记忆
                if conversation_id not in self.memories:
                    self.memories[conversation_id] = self._load_memory_from_file(conversation_id)
        else:
            # LLM存在时
            if conversation_id is None:
                # 生成新的conversation_id
                conversation_id = str(uuid.uuid4())
                logger.info(f"生成新的对话ID: {conversation_id}")
                # 为新ID创建空记忆
                self.memories[conversation_id] = ConversationBufferMemory()
                # 保存空记忆到文件
                self._save_memory_to_file(conversation_id, self.memories[conversation_id])
            else:
                # 确保对应conversation_id的记忆已加载
                if conversation_id not in self.memories:
                    self.memories[conversation_id] = self._load_memory_from_file(conversation_id)
        
        try:
            # 获取当前对话的记忆
            memory = self.memories[conversation_id]
            
            # 构建带记忆的完整提示
            memory_variables = memory.load_memory_variables({})
            full_prompt = ""
            if 'history' in memory_variables and memory_variables['history']:
                full_prompt += f"{memory_variables['history']}\n"
            full_prompt += f"Human: {prompt}\nAI: "
            
            # 将提示词输入LLM并获取回复
            response = self.llm.invoke(full_prompt)
            
            # 提取回复内容
            if hasattr(response, 'content'):
                reply = response.content
            elif isinstance(response, dict) and 'content' in response:
                reply = response['content']
            else:
                reply = str(response)
            
            # 确保回复内容不为空
            if not reply.strip():
                reply = "（暂无回复）"
            
            # 记录完整回复内容长度，用于调试
            logger.info(f"接收到的完整回复长度: {len(reply)} 字符")
            
            # 将新的对话内容保存到记忆中
            memory.save_context({'input': prompt}, {'output': reply})
            
            # 保存记忆到文件
            self._save_memory_to_file(conversation_id, memory)
            
            logger.info("成功获取LLM回复并更新记忆")
            return {
                'response': reply,
                'conversation_id': conversation_id
            }
        except Exception as e:
            logger.error(f"获取LLM回复时出错: {e}")
            return None


def create_agent_node(config_path: Optional[str] = None, model_name: str = "glm-4", provider: str = "zhipu", **kwargs) -> Optional[Callable[[Dict[str, Any]], Dict[str, Any]]]:
    """
    将Agent类初始化创建LLM后，包装成langgraph的一个图节点并返回
    
    Args:
        config_path (str, optional): 配置文件路径，默认为None
        model_name (str): 模型名称，默认为"glm-4"
        provider (str): LLM提供商，可选值: "zhipu"(智谱AI) 或 "openai"(通用OpenAI兼容接口)，默认为"zhipu"
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
        
        # 创建LLM，传递provider参数
        if not agent.create_llm(model_name=model_name, provider=provider, **kwargs):
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
                
                # 从状态中提取conversation_id
                conversation_id = state.get('conversation_id')
                
                # 使用Agent生成回复
                result = agent.generate_response(prompt, conversation_id)
                
                if result is not None:
                    # 更新状态字典
                    updated_state = {
                        **state,
                        'response': result['response'],
                        'agent_output': result['response'],
                        'conversation_id': result['conversation_id'],
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