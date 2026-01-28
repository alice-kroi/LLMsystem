import os
import yaml
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config_to_env(config_path="e:\\GitHub\\config.yaml", return_dict=False):
    """
    读取YAML配置文件并设置为环境变量
    
    Args:
        config_path (str): 配置文件的路径，如果为None则使用默认路径（LLM目录下的config.yaml）
        return_dict (bool): 是否返回解析后的配置字典，默认为False
        
    Returns:
        dict or None: 如果return_dict为True，则返回配置字典，否则返回None
    """
    try:
        # 如果未提供配置文件路径，使用默认路径
        if config_path is None:
            # 获取当前模块所在目录的绝对路径
            module_dir = os.path.dirname(os.path.abspath(__file__))
            # 计算LLM目录的路径（tool目录的上一级）
            llm_dir = os.path.dirname(module_dir)
            # 构建配置文件的绝对路径
            config_path = os.path.join(llm_dir, 'config.yaml')
        
        # 获取绝对路径，确保相对路径正确解析
        absolute_path = os.path.abspath(config_path)
        
        # 检查文件是否存在
        if not os.path.exists(absolute_path):
            logger.error(f"配置文件不存在: {absolute_path}")
            return None
        
        # 读取YAML文件
        with open(absolute_path, 'r', encoding='utf-8') as file:
            config_dict = yaml.safe_load(file)
        
        # 检查配置是否有效
        if config_dict is None:
            logger.warning(f"配置文件为空或格式无效: {absolute_path}")
            return config_dict if return_dict else None
        
        # 设置环境变量
        _set_env_variables(config_dict)
        
        logger.info(f"成功从{absolute_path}加载配置并设置环境变量")
        
        # 根据参数决定是否返回字典
        return config_dict if return_dict else None
        
    except yaml.YAMLError as e:
        logger.error(f"解析YAML文件时出错: {e}")
        return None
    except Exception as e:
        logger.error(f"加载配置时出错: {e}")
        return None

def _set_env_variables(config_dict, prefix=''):
    """
    递归地将配置字典中的所有键值对设置为环境变量
    
    Args:
        config_dict (dict): 配置字典
        prefix (str): 环境变量名的前缀
    """
    for key, value in config_dict.items():
        # 构建环境变量名（转为大写，使用下划线分隔）
        env_key = f"{prefix}{key.upper().replace('.', '_')}"
        
        if isinstance(value, dict):
            # 如果值是字典，递归处理
            _set_env_variables(value, f"{env_key}_")
        else:
            # 将值转换为字符串并设置为环境变量
            os.environ[env_key] = str(value)
            logger.debug(f"设置环境变量: {env_key}={value}")