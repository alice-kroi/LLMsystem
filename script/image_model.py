import os
import logging
import shutil
from typing import Optional, List, Union
from huggingface_hub import snapshot_download, hf_hub_download, login

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('image_model')


class ImageModelDownloader:
    """
    Hugging Face模型下载器，用于下载模型文件到本地目录
    """
    
    def __init__(self, cache_dir: Optional[str] = None):
        """
        初始化模型下载器
        
        Args:
            cache_dir: 缓存目录路径，如果不指定则使用huggingface_hub的默认缓存目录
        """
        self.cache_dir = cache_dir
    
    def login_huggingface(self, token: str) -> bool:
        """
        登录Hugging Face账号（用于访问需要认证的模型）
        
        Args:
            token: Hugging Face的访问令牌
            
        Returns:
            bool: 登录是否成功
        """
        try:
            login(token=token)
            logger.info("成功登录Hugging Face")
            return True
        except Exception as e:
            logger.error(f"登录Hugging Face失败: {e}")
            return False
    
    def download_model(self, 
                      model_id: str, 
                      local_dir: str, 
                      force_download: bool = False, 
                      resume_download: bool = True,
                      proxies: Optional[dict] = None,
                      token: Optional[str] = None,
                      allow_patterns: Optional[Union[str, List[str]]] = None,
                      ignore_patterns: Optional[Union[str, List[str]]] = None) -> bool:
        """
        下载完整的模型快照到本地目录
        
        Args:
            model_id: Hugging Face模型ID，格式为'用户名/模型名'
            local_dir: 本地保存目录路径
            force_download: 是否强制重新下载
            resume_download: 是否支持断点续传
            proxies: 代理设置
            token: Hugging Face访问令牌（可选）
            allow_patterns: 允许下载的文件模式，例如['*.safetensors', '*.bin', '*.json']
            ignore_patterns: 忽略的文件模式，例如['*.md', '*.txt', '*.git*']
            
        Returns:
            bool: 下载是否成功
        """
        try:
            # 检查并创建本地目录
            if not os.path.exists(local_dir):
                os.makedirs(local_dir, exist_ok=True)
                logger.info(f"创建本地目录: {local_dir}")
            
            # 如果强制下载且目录存在，清空目录
            if force_download and os.path.exists(local_dir):
                logger.info(f"强制重新下载，清空目录: {local_dir}")
                shutil.rmtree(local_dir)
                os.makedirs(local_dir, exist_ok=True)
            
            # 下载模型
            logger.info(f"开始下载模型 {model_id} 到 {local_dir}")
            snapshot_download(
                repo_id=model_id,
                local_dir=local_dir,
                cache_dir=self.cache_dir,
                resume_download=resume_download,
                proxies=proxies,
                token=token,
                allow_patterns=allow_patterns,
                ignore_patterns=ignore_patterns,
                # 确保下载的文件按照正确的路径结构保存
                local_dir_use_symlinks=False
            )
            
            logger.info(f"模型 {model_id} 下载完成")
            return True
            
        except Exception as e:
            logger.error(f"下载模型 {model_id} 失败: {e}")
            return False
    
    def download_specific_files(self, 
                              model_id: str, 
                              filenames: List[str], 
                              local_dir: str,
                              force_download: bool = False,
                              proxies: Optional[dict] = None,
                              token: Optional[str] = None) -> dict:
        """
        下载模型中的特定文件到本地目录
        
        Args:
            model_id: Hugging Face模型ID
            filenames: 要下载的文件名列表
            local_dir: 本地保存目录
            force_download: 是否强制重新下载
            proxies: 代理设置
            token: Hugging Face访问令牌
            
        Returns:
            dict: 包含每个文件下载状态的字典
        """
        results = {}
        
        # 检查并创建本地目录
        if not os.path.exists(local_dir):
            os.makedirs(local_dir, exist_ok=True)
            logger.info(f"创建本地目录: {local_dir}")
        
        # 下载每个指定的文件
        for filename in filenames:
            try:
                # 构建本地文件路径
                local_file_path = os.path.join(local_dir, filename)
                
                # 如果强制下载或文件不存在，则下载
                if force_download or not os.path.exists(local_file_path):
                    logger.info(f"下载文件 {filename} 从模型 {model_id}")
                    hf_hub_download(
                        repo_id=model_id,
                        filename=filename,
                        local_dir=local_dir,
                        cache_dir=self.cache_dir,
                        force_download=force_download,
                        proxies=proxies,
                        token=token
                    )
                    results[filename] = {
                        'status': 'success',
                        'path': local_file_path
                    }
                    logger.info(f"文件 {filename} 下载成功")
                else:
                    results[filename] = {
                        'status': 'skipped',
                        'path': local_file_path,
                        'reason': '文件已存在'
                    }
                    logger.info(f"文件 {filename} 已存在，跳过下载")
            
            except Exception as e:
                results[filename] = {
                    'status': 'failed',
                    'error': str(e)
                }
                logger.error(f"下载文件 {filename} 失败: {e}")
        
        return results
    
    def list_model_files(self, 
                        model_id: str,
                        token: Optional[str] = None,
                        proxies: Optional[dict] = None) -> Optional[List[str]]:
        """
        列出模型库中的所有文件
        
        Args:
            model_id: Hugging Face模型ID
            token: Hugging Face访问令牌
            proxies: 代理设置
            
        Returns:
            Optional[List[str]]: 文件列表，如果失败则返回None
        """
        try:
            from huggingface_hub import HfApi
            api = HfApi(token=token)
            files = api.list_repo_files(repo_id=model_id, repo_type="model", proxies=proxies)
            logger.info(f"成功获取模型 {model_id} 的文件列表，共 {len(files)} 个文件")
            return files
        except Exception as e:
            logger.error(f"获取模型 {model_id} 的文件列表失败: {e}")
            return None


def main():
    """
    示例用法
    """
    # 初始化下载器
    downloader = ImageModelDownloader()
    
    # 示例1：下载完整模型
    # 注意：这会下载所有文件，可能会非常大
    # downloader.download_model(
    #     model_id="runwayml/stable-diffusion-v1-5",
    #     local_dir="e:/models/stable-diffusion-v1-5",
    #     # 只下载必要的模型文件
    #     allow_patterns=["*.safetensors", "*.bin", "*.json", "tokenizer/*"],
    #     ignore_patterns=["*.md", "*.txt", ".git*"]
    # )
    
    # 示例2：下载特定文件
    # results = downloader.download_specific_files(
    #     model_id="runwayml/stable-diffusion-v1-5",
    #     filenames=["v1-5-pruned-emaonly.safetensors", "config.json"],
    #     local_dir="e:/models/stable-diffusion-v1-5"
    # )
    # print(results)
    
    # 示例3：列出模型文件
    # files = downloader.list_model_files(model_id="runwayml/stable-diffusion-v1-5")
    # if files:
    #     print("模型文件列表:")
    #     for file in files:
    #         print(f"- {file}")


if __name__ == "__main__":
    main()