import os
import sys
import logging
import threading
import time
import uuid
import json
import multiprocessing
from typing import Dict, List, Optional, Any
from queue import Queue, Empty
import asyncio
import websockets
import wave
import pyaudio
import requests
# 导入音频相关模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from tts import GPTSoVITSClient
from LLM_base.Agent import Agent
from LLM_base.MilvusRAG import MilvusRAG, DoubaoEmbeddings
from LLM_base.prompt import load_prompt
from tool.config_load import load_config_to_env
# 设置环境变量并返回字典
 # 输出解析后的配置字典
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("VTuberSystem")
SERVER_URL = "http://localhost:8888"
# 设置环境变量并返回字典

def send_request(function, duration=0, params=None):
    """发送请求到Live2D控制器
    
    参数:
        function (str): 功能名称（"回答问题"、"有事件"、"无事件"）
        duration (int): 持续时间（秒）
        params (dict): 其他参数
        
    返回:
        dict: 服务器响应
    """
    if params is None:
        params = {}
    
    # 构建请求数据
    request_data = {
        "function": function,
        "time": duration,
        "params": params
    }
    
    try:
        # 发送POST请求
        response = requests.post(SERVER_URL, json=request_data)
        # 解析响应
        return response.json()
    except Exception as e:
        return {"status": "error", "message": str(e)}

def play_audio_external(audio_path):
    """在外部进程中播放音频文件"""
    try:
        # 打开音频文件
        wf = wave.open(audio_path, 'rb')
        
        # 初始化PyAudio
        p = pyaudio.PyAudio()
        send_request("回答问题", duration=wf.getnframes() / wf.getframerate())
        # 定义回调函数
        def callback(in_data, frame_count, time_info, status):
            data = wf.readframes(frame_count)
            return (data, pyaudio.paContinue)
        
        # 打开音频流
        stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                       channels=wf.getnchannels(),
                       rate=wf.getframerate(),
                       output=True,
                       stream_callback=callback)
        
        # 开始播放
        stream.start_stream()
        
        # 等待播放完成
        while stream.is_active():
            time.sleep(0.1)
        
        # 停止并关闭流
        stream.stop_stream()
        stream.close()
        
        # 关闭PyAudio
        p.terminate()
        wf.close()
        
        logger.info(f"音频播放完成: {audio_path}")
    except Exception as e:
        logger.error(f"音频播放失败: {e}")

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
        
        # TTS配置
        self.tts_client = GPTSoVITSClient(api_url="http://127.0.0.1:9880")
        self.audio_output_dir = "temp_audio"
        os.makedirs(self.audio_output_dir, exist_ok=True)
        # 情感对应的参考音频路径
        self.emotion_audio_paths = {
            "调皮": "E:\【GPT-SoVITS】爱莉希雅V2\参考音频\【调皮】哇已经发展成三人关系了吗？芽衣你真是越来越大胆了呢。.wav",
            "调侃": "E:\【GPT-SoVITS】爱莉希雅V2\参考音频\【调侃的失望】啊真是的，头也不回的走掉了呢。.wav",
            "尴尬": "E:\【GPT-SoVITS】爱莉希雅V2\参考音频\【尴尬】呃，芽衣，你的问题还真是一如既往的，刁钻。.wav",
            "感动": "E:\【GPT-SoVITS】爱莉希雅V2\参考音频\【感动】能以这种方式见到你，我真的好幸运，你带给了我，一直渴望却又不敢奢求的礼物。.wav",
            "积极": "E:\【GPT-SoVITS】爱莉希雅V2\参考音频\【积极】执拗的花朵永远不会因暴雨而褪去颜色，你的决心也一定能在绝境中绽放真我。.wav",
            "急了": "E:\【GPT-SoVITS】爱莉希雅V2\参考音频\【急了】啊等等，难道说背叛者指的是芽衣的事，千万别这样想呀，我心里还是有你的。.wav",
            "假装": "E:\【GPT-SoVITS】爱莉希雅V2\参考音频\【假装】拜托了医生，对我来说这真的很重要。.wav",
            "惊喜": "E:\【GPT-SoVITS】爱莉希雅V2\参考音频\【惊喜】哇，那不是预约不知排到什么时候的超级餐厅嘛，突然带个人会不会给你添麻烦呀？.wav",
            "开心": "E:\【GPT-SoVITS】爱莉希雅V2\参考音频\【开心】哎呀，看到美少女突然来访，比起惊讶，要表现的更开心一些才行啊。.wav",
            "撩拨": "E:\【GPT-SoVITS】爱莉希雅V2\参考音频\【撩拨】哎呀，我还以为你会好好记住人家的名字的，有点难过。.wav",
            "难过": "E:\【GPT-SoVITS】爱莉希雅V2\参考音频\【难过】对你来说，对任何人来说，我们，意味着什么呢？.wav",
            "疲惫": "E:\【GPT-SoVITS】爱莉希雅V2\参考音频\【疲惫】我知道你在想什么，不过也稍微休息一下吧。.wav",
            "普通": "E:\【GPT-SoVITS】爱莉希雅V2\参考音频\【普通】爱莉希雅的贴心提示，你可以尽情依赖爱莉希雅，而她也会以全部的真心来回应你。.wav",
            "撒娇": "E:\【GPT-SoVITS】爱莉希雅V2\参考音频\【撒娇】我今天一定要知道这个，不然哪儿都不让你去，你就告诉我吧告诉我嘛，好不好？.wav",
            "生气": "E:\【GPT-SoVITS】爱莉希雅V2\参考音频\【生气】记错了，那不是他，为什么唯独对这件事印象这么深刻？.wav",
            "严肃": "E:\【GPT-SoVITS】爱莉希雅V2\参考音频\【严肃】不对，既然他已经给出了判断，你准备去做什么也没那么重要了。.wav",
            "疑问": "E:\【GPT-SoVITS】爱莉希雅V2\参考音频\【疑问】白纸，白纸，哪里能找到呢？.wav",
            "自言": "E:\【GPT-SoVITS】爱莉希雅V2\参考音频\【自言】毕竟，我这一次又是来请他帮忙的，被他听到，恐怕要了不得了呢。.wav",
        }
        
        self._initialize_system()
    
    def _initialize_system(self):
        
        try:
            self.vtuber_character_prompt = load_prompt("vtuber_character")
            if self.vtuber_character_prompt is None:
                logger.error("无法加载VTuber角色设定提示词")
                raise ValueError("VTuber角色设定提示词加载失败")
            print(self.config_path)
            self.agent = Agent(config_path=self.config_path)
            if not self.agent.create_llm():
                logger.error("无法创建LLM实例")
                raise RuntimeError("LLM创建失败")
            # 初始化 MilvusRAG 实例
            self.rag = MilvusRAG(
                uri="http://localhost:19530",
                token="root:Milvus",
                dbname="vtuber",
                embedding_model=DoubaoEmbeddings()
            )
            
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
        
        # 启动WebSocket服务器
        self.start_websocket()
    
    def stop(self):
        self.running = False
        if hasattr(self.rag, 'close'):
            self.rag.close()
    
    def add_message(self, user_id: str, username: str, content: str):
        message = VTuberMessage(user_id, username, content)
        self.message_queue.put(message)
        
        # 将用户消息添加到 MilvusRAG 中
        try:
            self.rag.add_user_message(user_id, username, content)
        except Exception as e:
            logger.error(f"添加用户消息到 MilvusRAG 时出错: {e}")
    
    def _process_messages(self):
        while self.running:
            try:
                message = self.message_queue.get(timeout=1)
                self._process_single_message(message)
                self.message_queue.task_done()
            except Empty:
                continue
            except Exception as e:
                logger.error(f"处理消息队列时出错: {e}")
    
    def _process_single_message(self, message: VTuberMessage):
        try:
            relevant_info = self._retrieve_relevant_info(message.content)
            formatted_prompt = self._format_prompt(message, relevant_info)
            conversation_id = self._get_conversation_id(message.user_id)
            response = self._generate_response(formatted_prompt, conversation_id)
            self._record_conversation(message, response, conversation_id)
            
            # 将AI回复添加到 MilvusRAG 中
            try:
                self.rag.add_llm_response(message.user_id, message.username, response)
            except Exception as e:
                logger.error(f"添加AI回复到 MilvusRAG 时出错: {e}")
                
            logger.info(f"{message.username}: {message.content}")
            logger.info(f"星野梦咲: {response}")
            print(response)
            # 生成并播放音频
            self._generate_and_play_audio(response)
            
        except Exception as e:
            logger.error(f"处理留言时出错: {e}")
    
    def _retrieve_relevant_info(self, query: str) -> str:
        try:
            # 使用 MilvusRAG 的 semantic_similarity_search 方法
            results = self.rag.semantic_similarity_search(query, top_k=3)
            if results:
                return "\n".join([result["content"] for result in results])
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
    
    def _generate_and_play_audio(self, response: str):
        """
        生成音频并播放
        
        Args:
            response (str): 包含情感标记和回复内容的字符串
        """
        try:
            
            # 提取情感标记，格式如【开心】
            if response.startswith("【") and "】" in response:
                emotion_end = response.index("】")
                emotion = response[1:emotion_end]
                content = response[emotion_end+1:].strip()
            else:
                emotion = "普通"
                content = response.strip()
            
            # 获取对应的参考音频
            ref_audio_path = self.emotion_audio_paths.get(emotion, self.emotion_audio_paths["普通"])
            print(content)
            print(ref_audio_path)
            # 生成唯一的输出文件名
            output_filename = f"response_{int(time.time())}_{uuid.uuid4().hex[:8]}.wav"
            output_path = os.path.join(self.audio_output_dir, output_filename)
            print([
                content,
                "zh",
                ref_audio_path,
                output_path,
                "zh",
                "",
            ])
            # 生成音频
            success = self.tts_client.generate_audio(
                text=content,
                text_lang="zh",
                ref_audio_path=ref_audio_path,
                output_path=output_path,
                prompt_lang="zh",
                prompt_text=""
            )
            
            if success:
                # 在新进程中播放音频
                self._play_audio_in_process(output_path)
            else:
                logger.error("生成音频失败")
                
        except Exception as e:
            logger.error(f"处理音频时出错: {e}")
    
    def _play_audio_in_process(self, audio_path: str):
        """
        在新进程中播放音频
        
        Args:
            audio_path (str): 音频文件路径
        """

        # 创建并启动新进程
        audio_process = multiprocessing.Process(target=play_audio_external, args=(audio_path,))
        audio_process.daemon = True
        audio_process.start()
    
    # WebSocket核心功能
    async def _websocket_handler(self, websocket):
        try:
            async for message in websocket:
                data = json.loads(message)
                user_id = data.get("user").get('uid', 'anonymous')
                username = data.get("user").get('uname', '匿名用户')
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
            
            # 将消息添加到 MilvusRAG 中
            try:
                self.rag.add_user_message(user_id, username, content)
                self.rag.add_llm_response(user_id, username, response)
            except Exception as e:
                logger.error(f"添加消息到 MilvusRAG 时出错: {e}")
            
            # 生成并播放音频
            self._generate_and_play_audio(response)
            
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
        
        if hasattr(self, 'ws_thread') and self.ws_thread and self.ws_thread.is_alive():
            return
        
        self.ws_thread = threading.Thread(target=run_server)
        self.ws_thread.daemon = True
        self.ws_thread.start()

# 导入提示词加载函数
from LLM_base.prompt import load_prompt

def main():
    try:
        vtuber = VTuberSystem(ws_port=8765,config_path="e:\\GitHub\\config.yaml")
        vtuber.start()
        
        print(f"WebSocket已启动，监听端口: {vtuber.ws_port}")
        print("输入 'quit' 停止服务\n")
        
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