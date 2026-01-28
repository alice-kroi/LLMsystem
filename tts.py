import requests
import os
import json
from typing import Optional, List

class GPTSoVITSClient:
    """
    GPT-SoVITS-v2pro API客户端，用于发送文本转语音请求
    """
    
    def __init__(self, api_url: str = "http://127.0.0.1:9880"):
        """
        初始化客户端
        
        参数:
            api_url: API服务地址，默认为 http://127.0.0.1:9880
        """
        self.api_url = api_url.rstrip("/")
        self.tts_endpoint = f"{self.api_url}/tts"
        
    def generate_audio(
        self,
        text: str,
        text_lang: str,
        ref_audio_path: str,
        output_path: str,
        prompt_lang: str,
        prompt_text: str = "",
        aux_ref_audio_paths: Optional[List[str]] = None,
        top_k: int = 5,
        top_p: float = 1.0,
        temperature: float = 1.0,
        text_split_method: str = "cut5",
        batch_size: int = 1,
        batch_threshold: float = 0.75,
        split_bucket: bool = True,
        speed_factor: float = 1.0,
        fragment_interval: float = 0.3,
        seed: int = -1,
        media_type: str = "wav",
        streaming_mode: bool = False,
        parallel_infer: bool = True,
        repetition_penalty: float = 1.35,
        sample_steps: int = 32,
        super_sampling: bool = False
    ) -> bool:
        """
        发送文本转语音请求并保存音频
        
        参数:
            text: 需要转换的文本
            text_lang: 文本语言（如"zh", "en", "ja"等）
            ref_audio_path: 参考音频路径，用于音色克隆
            output_path: 生成的音频保存路径
            prompt_lang: 参考音频的语言
            prompt_text: 参考音频对应的文本（可选）
            aux_ref_audio_paths: 辅助参考音频路径列表（可选）
            top_k: 采样参数top_k
            top_p: 采样参数top_p
            temperature: 采样温度
            text_split_method: 文本分割方法
            batch_size: 批处理大小
            batch_threshold: 批处理阈值
            split_bucket: 是否使用分桶策略
            speed_factor: 语速因子
            fragment_interval: 片段间隔
            seed: 随机种子
            media_type: 输出音频格式
            streaming_mode: 是否使用流式传输
            parallel_infer: 是否使用并行推理
            repetition_penalty: 重复惩罚
            sample_steps: 采样步数
            super_sampling: 是否使用超采样
            
        返回:
            bool: 成功返回True，失败返回False
        """
        
        # 构建请求参数
        payload = {
            "text": text,
            "text_lang": text_lang,
            "ref_audio_path": ref_audio_path,
            "prompt_lang": prompt_lang,
            "prompt_text": prompt_text,
            "aux_ref_audio_paths": aux_ref_audio_paths or [],
            "top_k": top_k,
            "top_p": top_p,
            "temperature": temperature,
            "text_split_method": text_split_method,
            "batch_size": batch_size,
            "batch_threshold": batch_threshold,
            "split_bucket": split_bucket,
            "speed_factor": speed_factor,
            "fragment_interval": fragment_interval,
            "seed": seed,
            "media_type": media_type,
            "streaming_mode": streaming_mode,
            "parallel_infer": parallel_infer,
            "repetition_penalty": repetition_penalty,
            "sample_steps": sample_steps,
            "super_sampling": super_sampling
        }
        
        try:
            # 发送POST请求
            print(f"正在发送TTS请求...")
            response = requests.post(self.tts_endpoint, json=payload)
            
            # 检查响应状态码
            if response.status_code == 200:
                # 确保输出目录存在
                output_dir = os.path.dirname(output_path)
                if output_dir and not os.path.exists(output_dir):
                    os.makedirs(output_dir, exist_ok=True)
                
                # 保存音频文件
                with open(output_path, "wb") as f:
                    f.write(response.content)
                
                print(f"音频已成功生成并保存到: {output_path}")
                return True
            else:
                # 处理错误响应
                try:
                    error_data = response.json()
                    print(f"请求失败: {response.status_code}")
                    print(f"错误信息: {error_data.get('message', '未知错误')}")
                except json.JSONDecodeError:
                    print(f"请求失败: {response.status_code}")
                    print(f"响应内容: {response.text}")
                return False
                
        except requests.exceptions.RequestException as e:
            print(f"请求发生异常: {e}")
            return False
        except Exception as e:
            print(f"发生未知异常: {e}")
            return False

# 使用示例
if __name__ == "__main__":
    # 创建客户端实例
    client = GPTSoVITSClient(api_url="http://127.0.0.1:9880")
    
    # 设置参数
    text = "星光你好呀！我是来自星之次元的星野梦咲的说~很高兴认识你哦！"
    text_lang = "zh"
    ref_audio_path = "E:\\【GPT-SoVITS】爱莉希雅V2\\参考音频\\【调皮】哇已经发展成三人关系了吗？芽衣你真是越来越大胆了呢。.wav"  # 替换为你的参考音频路径
    output_path = "temp_audio\\response_1766144967_e8c0498f.wav"  # 输出音频路径
    prompt_lang = "zh"
    prompt_text = "哇已经发展成三人关系了吗？芽衣你真是越来越大胆了呢。"  # 如果有的话
    
    # 发送请求
    success = client.generate_audio(
        text=text,
        text_lang=text_lang,
        ref_audio_path=ref_audio_path,
        output_path=output_path,
        prompt_lang=prompt_lang,
        prompt_text=prompt_text
    )
    
    if success:
        print("任务完成！")
    else:
        print("任务失败！")