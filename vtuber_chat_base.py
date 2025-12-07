import os
import sys
import logging
import threading
import time
import uuid
from typing import Dict, List, Optional, Any
from queue import Queue, Empty

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入所需模块
from LLM_base.Agent import Agent
from LLM_base.RAG import RAG
from LLM_base.prompt import load_prompt, get_prompt

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("VTuberSystem")

class VTuberMessage:
    """
    VTuber消息类，用于封装观众的留言
    """
    def __init__(self, user_id: str, username: str, content: str):
        self.user_id = user_id
        self.username = username
        self.content = content
        self.timestamp = time.time()
        self.message_id = str(uuid.uuid4())

class VTuberSystem:
    """
    VTuber系统类，实现具有长期记忆功能的VTuber
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化VTuber系统
        
        Args:
            config_path: 配置文件路径
        """
        self.config_path = config_path
        self.agent = None
        self.rag = None
        self.message_queue = Queue()
        self.processing_thread = None
        self.running = False
        self.vtuber_character_prompt = None
        self.conversation_memory = {}
        
        # 初始化系统
        self._initialize_system()
    
    def _initialize_system(self):
        """
        初始化系统组件
        """
        try:
            # 加载VTuber角色设定提示词
            self.vtuber_character_prompt = load_prompt("vtuber_character")
            if self.vtuber_character_prompt is None:
                logger.error("无法加载VTuber角色设定提示词")
                raise ValueError("VTuber角色设定提示词加载失败")
            logger.info("成功加载VTuber角色设定提示词")
            
            # 初始化Agent
            self.agent = Agent(config_path=self.config_path)
            logger.info("成功初始化Agent")
            
            # 创建LLM
            if not self.agent.create_llm():
                logger.error("无法创建LLM实例")
                raise RuntimeError("LLM创建失败")
            logger.info("成功创建LLM实例")
            
            # 初始化RAG系统
            self.rag = RAG()
            logger.info("成功初始化RAG系统")
            
        except Exception as e:
            logger.error(f"初始化系统时出错: {e}")
            raise
    
    def start(self):
        """
        启动VTuber系统
        """
        if self.running:
            logger.warning("VTuber系统已经在运行")
            return
        
        self.running = True
        
        # 启动消息处理线程
        self.processing_thread = threading.Thread(target=self._process_messages)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        logger.info("VTuber系统已启动")
    
    def stop(self):
        """
        停止VTuber系统
        """
        if not self.running:
            logger.warning("VTuber系统已经停止")
            return
        
        self.running = False
        
        # 等待消息处理线程结束
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(5)
        
        logger.info("VTuber系统已停止")
    
    def send_message(self, user_id: str, username: str, content: str) -> str:
        """
        发送观众留言到VTuber系统
        
        Args:
            user_id: 用户ID
            username: 用户名
            content: 留言内容
            
        Returns:
            消息ID
        """
        message = VTuberMessage(user_id, username, content)
        self.message_queue.put(message)
        logger.info(f"收到新消息: {message.message_id} - {username}: {content[:50]}...")
        return message.message_id
    
    def _process_messages(self):
        """
        处理消息队列中的观众留言
        """
        while self.running:
            try:
                # 从队列中获取消息，超时1秒
                message = self.message_queue.get(timeout=1)
                
                # 处理消息
                self._handle_message(message)
                
                # 标记消息处理完成
                self.message_queue.task_done()
                
            except Empty:
                continue
            except Exception as e:
                logger.error(f"处理消息时出错: {e}")
    
    def _handle_message(self, message: VTuberMessage):
        """
        处理单个观众留言
        
        Args:
            message: 观众留言对象
        """
        try:
            # 从RAG检索相关信息
            relevant_info = self._retrieve_relevant_info(message.content)
            
            # 格式化提示词
            formatted_prompt = self._format_prompt(message, relevant_info)
            
            # 获取或创建对话ID
            conversation_id = self._get_conversation_id(message.user_id)
            
            # 生成回复
            response = self._generate_response(formatted_prompt, conversation_id)
            
            # 记录对话
            self._record_conversation(message, response, conversation_id)
            
            # 输出回复
            logger.info(f"VTuber回复: {response[:50]}...")
            print(f"\n{message.username}: {message.content}")
            print(f"VTuber: {response}")
            print("=" * 50)
            
        except Exception as e:
            logger.error(f"处理留言时出错: {e}")
    
    def _retrieve_relevant_info(self, query: str) -> str:
        """
        从RAG检索与查询相关的信息
        
        Args:
            query: 查询内容
            
        Returns:
            相关信息的字符串
        """
        try:
            # 使用RAG检索信息
            # 注意：这里需要根据RAG类的实际方法进行调整
            # 假设RAG类有一个retrieve方法
            results = self.rag.retrieve(query, top_k=3)
            
            if results:
                relevant_info = "\n".join([result.page_content for result in results])
                return relevant_info
            else:
                return ""
                
        except Exception as e:
            logger.error(f"检索相关信息时出错: {e}")
            return ""
    
    def _format_prompt(self, message: VTuberMessage, relevant_info: str) -> str:
        """
        格式化提示词
        
        Args:
            message: 观众留言
            relevant_info: 相关信息
            
        Returns:
            格式化后的提示词
        """
        context = ""
        if relevant_info:
            context = f"\n\n【相关信息参考】\n{relevant_info}\n"
        
        return f"{self.vtuber_character_prompt}{context}\n\n{message.username}：{message.content}\n\n星野梦咲："
    
    def _get_conversation_id(self, user_id: str) -> str:
        """
        获取或创建用户的对话ID
        
        Args:
            user_id: 用户ID
            
        Returns:
            对话ID
        """
        if user_id not in self.conversation_memory:
            # 为新用户创建对话ID
            self.conversation_memory[user_id] = str(uuid.uuid4())
        
        return self.conversation_memory[user_id]
    
    def _generate_response(self, prompt: str, conversation_id: str) -> str:
        """
        生成VTuber回复
        
        Args:
            prompt: 提示词
            conversation_id: 对话ID
            
        Returns:
            VTuber回复内容
        """
        try:
            # 使用Agent生成回复
            result = self.agent.generate_response(prompt, conversation_id=conversation_id)
            
            if result:
                return result['response']
            else:
                logger.error("无法生成回复")
                return "抱歉，我现在有点忙，稍后再和你聊吧~"
                
        except Exception as e:
            logger.error(f"生成回复时出错: {e}")
            return "哎呀，刚才发生了一点小问题，我们换个话题聊聊吧~"
    
    def _record_conversation(self, message: VTuberMessage, response: str, conversation_id: str):
        """
        记录对话历史
        
        Args:
            message: 观众留言
            response: VTuber回复
            conversation_id: 对话ID
        """
        # 对话历史已由Agent类自动管理和保存，这里可以添加额外的记录逻辑
        logger.info(f"记录对话: {conversation_id} - {message.username} -> VTuber")

# 示例使用
def main():
    """
    VTuber系统示例使用
    """
    try:
        # 初始化VTuber系统
        vtuber = VTuberSystem()
        
        # 启动系统
        vtuber.start()
        
        print("🌟 欢迎来到星野梦咲的直播间！")
        print("我是星野梦咲，来自星之次元的虚拟主播~ ✨")
        print("输入 '退出' 或 'quit' 可以结束聊天哦~\n")
        
        # 模拟观众留言
        sample_messages = [
            VTuberMessage("user1", "星光1号", "你好呀，梦咲！今天过得怎么样？"),
            VTuberMessage("user2", "星光2号", "梦咲喜欢吃什么食物呢？"),
            VTuberMessage("user1", "星光1号", "刚才你说喜欢甜点，能推荐几种好吃的吗？"),
            VTuberMessage("user3", "星光3号", "梦咲有没有看过最近很火的动漫呀？")
        ]
        
        # 发送示例消息
        for msg in sample_messages:
            vtuber.send_message(msg.user_id, msg.username, msg.content)
            time.sleep(1)  # 间隔1秒发送一条消息
        
        # 交互式聊天
        while True:
            user_input = input("你: ")
            
            if user_input.strip().lower() in ['退出', 'quit', 'exit']:
                break
            
            if not user_input.strip():
                continue
            
            # 发送用户输入作为观众留言
            vtuber.send_message("interactive_user", "互动用户", user_input)
        
        # 停止系统
        vtuber.stop()
        print("\n星野梦咲：感谢你的陪伴！下次再见~ 🌟")
        
    except Exception as e:
        logger.error(f"VTuber系统运行出错: {e}")
        print("系统运行出错，请查看日志了解详情")

if __name__ == "__main__":
    main()