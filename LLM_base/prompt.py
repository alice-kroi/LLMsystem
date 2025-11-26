import os
import logging
from typing import Optional

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 提示词目录路径
PROMPT_DIR = os.path.join(os.path.dirname(__file__), 'prompt')

class PromptLoader:
    """
    提示词加载器，负责从文件中加载预定义的提示词模板
    """
    
    def __init__(self, prompt_dir: str = PROMPT_DIR):
        """
        初始化提示词加载器
        
        Args:
            prompt_dir: 提示词文件所在目录路径
        """
        self.prompt_dir = prompt_dir
        logger.info(f"初始化提示词加载器，提示词目录: {self.prompt_dir}")
    
    def load_prompt(self, prompt_name: str) -> Optional[str]:
        """
        加载指定名称的提示词模板
        
        Args:
            prompt_name: 提示词模板名称，不带文件扩展名
        
        Returns:
            加载的提示词模板内容，如果文件不存在或读取失败则返回None
        """
        try:
            # 构建提示词文件路径
            prompt_file = os.path.join(self.prompt_dir, f"{prompt_name}.txt")
            
            # 检查文件是否存在
            if not os.path.exists(prompt_file):
                logger.error(f"提示词文件不存在: {prompt_file}")
                return None
            
            # 读取提示词内容
            with open(prompt_file, 'r', encoding='utf-8') as f:
                prompt_content = f.read().strip()
            
            logger.info(f"成功加载提示词模板: {prompt_name}")
            return prompt_content
            
        except Exception as e:
            logger.error(f"加载提示词模板 '{prompt_name}' 失败: {str(e)}")
            return None
    
    def get_formatted_prompt(self, prompt_name: str, user_input: str) -> Optional[str]:
        """
        获取格式化后的提示词，将用户输入插入到模板中
        
        Args:
            prompt_name: 提示词模板名称
            user_input: 用户输入内容
            
        Returns:
            格式化后的完整提示词，如果加载失败则返回None
        """
        # 加载模板
        template = self.load_prompt(prompt_name)
        if template is None:
            return None
        
        # 格式化提示词
        try:
            formatted_prompt = template.replace("{user_input}", user_input)
            logger.debug(f"成功格式化提示词: {prompt_name}")
            return formatted_prompt
        except Exception as e:
            logger.error(f"格式化提示词失败: {str(e)}")
            return None

# 创建全局提示词加载器实例
prompt_loader = PromptLoader()

# 便捷函数
def load_prompt(prompt_name: str) -> Optional[str]:
    """
    便捷函数：加载指定名称的提示词模板
    
    Args:
        prompt_name: 提示词模板名称
        
    Returns:
        加载的提示词模板内容
    """
    return prompt_loader.load_prompt(prompt_name)

def get_prompt(prompt_name: str, user_input: str) -> Optional[str]:
    """
    便捷函数：获取格式化后的完整提示词
    
    Args:
        prompt_name: 提示词模板名称
        user_input: 用户输入内容
        
    Returns:
        格式化后的完整提示词
    """
    return prompt_loader.get_formatted_prompt(prompt_name, user_input)

# 预定义的提示词名称常量
CONTRACT_GENERATION_PROMPT = "contract_generation"
CONTRACT_REVIEW_PROMPT = "contract_review"
CUSTOMER_SERVICE_PROMPT = "customer_service"