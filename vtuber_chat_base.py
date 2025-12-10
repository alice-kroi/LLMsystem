import os
import sys
import logging
import threading
import time
import uuid
import json
from typing import Dict, List, Optional, Any
from queue import Queue, Empty
import asyncio
import websockets

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from LLM_base.Agent import Agent
from LLM_base.RAG import RAG
from LLM_base.prompt import load_prompt, get_prompt

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("VTuberSystem")

class VTuberMessage:
    def __init__(self, user_id: str, username: str, content: str):
        self.user_id = user_id
        self.username = username
        self.content = content
        self.timestamp = time.time()
        self.message_id = str(uuid.uuid4())

class VTuberSystem:
    def __init__(self, config_path: Optional[str] = None, ws_port: int = 8765):
        self.config_path = config_path
        self.agent = None
        self.rag = None
        self.message_queue = Queue()
        self.processing_thread = None
        self.running = False
        self.vtuber_character_prompt = None
        self.conversation_memory = {}
        
        # WebSocket配置
        self.ws_port = ws_port
        self.ws_thread = None
        
        self._initialize_system()
    
    def _initialize_system(self):
        try:
            self.vtuber_character_prompt = load_prompt("vtuber_character")
            if self.vtuber_character_prompt is None:
                logger.error("无法加载VTuber角色设定提示词")
                raise ValueError("VTuber角色设定提示词加载失败")
            
            self.agent = Agent(config_path=self.config_path)
            if not self.agent.create_llm():
                logger.error("无法创建LLM实例")
                raise RuntimeError("LLM创建失败")
            
            self.rag = RAG()
            
        except Exception as e:
            logger.error(f"初始化系统时出错: {e}")
            raise
    
    def start(self):
        if self.running:
            return
        
        self.running = True
        self.processing_thread = threading.Thread(target=self._process_messages)
        self.processing_thread.daemon = True
        self.processing_thread.start()
    
    def stop(self):
        if not self.running:
            return
        
        self.running = False
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(5)
    
    def send_message(self, user_id: str, username: str, content: str) -> str:
        message = VTuberMessage(user_id, username, content)
        self.message_queue.put(message)
        return message.message_id
    
    def _process_messages(self):
        while self.running:
            try:
                message = self.message_queue.get(timeout=1)
                self._handle_message(message)
                self.message_queue.task_done()
            except Empty:
                continue
            except Exception as e:
                logger.error(f"处理消息时出错: {e}")
    
    def _handle_message(self, message: VTuberMessage):
        try:
            relevant_info = self._retrieve_relevant_info(message.content)
            formatted_prompt = self._format_prompt(message, relevant_info)
            conversation_id = self._get_conversation_id(message.user_id)
            response = self._generate_response(formatted_prompt, conversation_id)
            self._record_conversation(message, response, conversation_id)
            
            print(f"\n{message.username}: {message.content}")
            print(f"VTuber: {response}")
            print("=" * 50)
            
        except Exception as e:
            logger.error(f"处理留言时出错: {e}")
    
    def _retrieve_relevant_info(self, query: str) -> str:
        try:
            results = self.rag.retrieve(query, top_k=3)
            if results:
                return "\n".join([result.page_content for result in results])
            return ""
        except Exception as e:
            logger.error(f"检索相关信息时出错: {e}")
            return ""
    
    def _format_prompt(self, message: VTuberMessage, relevant_info: str) -> str:
        context = f"\n\n【相关信息参考】\n{relevant_info}\n" if relevant_info else ""
        return f"{self.vtuber_character_prompt}{context}\n\n{message.username}：{message.content}\n\n星野梦咲："
    
    def _get_conversation_id(self, user_id: str) -> str:
        if user_id not in self.conversation_memory:
            self.conversation_memory[user_id] = str(uuid.uuid4())
        return self.conversation_memory[user_id]
    
    def _generate_response(self, prompt: str, conversation_id: str) -> str:
        try:
            result = self.agent.generate_response(prompt, conversation_id=conversation_id)
            if result:
                return result['response']
            return "抱歉，我现在有点忙，稍后再和你聊吧~"
        except Exception as e:
            logger.error(f"生成回复时出错: {e}")
            return "哎呀，刚才发生了一点小问题，我们换个话题聊聊吧~"
    
    def _record_conversation(self, message: VTuberMessage, response: str, conversation_id: str):
        logger.info(f"记录对话: {conversation_id} - {message.username} -> VTuber")
    
    # WebSocket核心功能
    async def _websocket_handler(self, websocket):
        try:
            async for message in websocket:
                data = json.loads(message)
                user_id = data.get('user_id', 'anonymous')
                username = data.get('username', '匿名用户')
                content = data.get('content', '')
                
                if not content:
                    await websocket.send(json.dumps({'status': 'error', 'message': '内容不能为空'}))
                    continue
                
                # 处理消息并生成回复
                response = await self._process_ws_message(user_id, username, content)
                
                # 返回JSON格式的回复
                await websocket.send(json.dumps({'status': 'success', 'response': response}))
                
        except json.JSONDecodeError:
            await websocket.send(json.dumps({'status': 'error', 'message': '无效的JSON格式'}))
        except Exception as e:
            logger.error(f"WebSocket错误: {e}")
    
    async def _process_ws_message(self, user_id: str, username: str, content: str) -> str:
        try:
            relevant_info = self._retrieve_relevant_info(content)
            message = VTuberMessage(user_id, username, content)
            formatted_prompt = self._format_prompt(message, relevant_info)
            conversation_id = self._get_conversation_id(user_id)
            response = self._generate_response(formatted_prompt, conversation_id)
            self._record_conversation(message, response, conversation_id)
            
            return response
        except Exception as e:
            logger.error(f"处理WS消息出错: {e}")
            return "处理消息时发生错误"
    
    def start_websocket(self):
        def run_server():
            # 创建并设置事件循环
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            async def main_coroutine():
                # 使用上下文管理器创建WebSocket服务器
                async with websockets.serve(self._websocket_handler, "localhost", self.ws_port):
                    logger.info(f"WebSocket监听端口: {self.ws_port}")
                    # 保持服务器运行（永久阻塞）
                    await asyncio.Future()
            
            try:
                # 在事件循环中运行主协程
                loop.run_until_complete(main_coroutine())
            except Exception as e:
                logger.error(f"WebSocket服务器错误: {e}")
            finally:
                loop.close()
        
        if self.ws_thread and self.ws_thread.is_alive():
            return
        
        self.ws_thread = threading.Thread(target=run_server)
        self.ws_thread.daemon = True
        self.ws_thread.start()

def main():
    try:
        vtuber = VTuberSystem(ws_port=8765)
        vtuber.start()
        vtuber.start_websocket()
        
        print(f"WebSocket已启动，监听端口: {vtuber.ws_port}")
        print("输入 'quit' 停止服务\\n")
        
        while True:
            user_input = input()
            if user_input.strip().lower() == 'quit':
                break
        
        vtuber.stop()
        print("服务已停止")
        
    except Exception as e:
        logger.error(f"运行错误: {e}")
        print("系统运行出错")

if __name__ == "__main__":
    main()