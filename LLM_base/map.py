import os
import sys
import logging
from typing import Dict, Any, Optional

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 现在可以正确导入tool模块
from tool.config_load import load_config_to_env
from langgraph.graph import StateGraph,MessageGraph

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LLMMap:
    def __init__(self, config_path=None):
        """
        初始化LLMMap类
        
        Args:
            config_path (str, optional): 配置文件路径，默认为None
        """
        # 利用config_load函数加载配置并存储
        self.config = load_config_to_env(config_path=config_path, return_dict=True)
        if self.config:
            logger.info("配置文件加载成功")
        else:
            logger.warning("配置文件加载失败或为空")
        
        # 创建空的map属性
        self.map = None
        logger.info("初始化空map完成")
    
    def set_map(self, graph=None, **kwargs):
        """
        设置或创建map
        
        Args:
            graph: 图对象，如果为None则创建空map
            **kwargs: 可选参数
        
        Returns:
            bool: 设置是否成功
        """
        try:
            # 检查输入参数图是否存在且符合格式
            if graph is not None:
                # 检查graph是否为Graph类型或兼容类型
                if isinstance(graph, StateGraph) or hasattr(graph, 'nodes') and hasattr(graph, 'edges'):
                    self.map = graph
                    logger.info("成功将输入图覆盖为map")
                    return True
                else:
                    logger.error("输入的graph不符合要求的格式")
                    return False
            else:
                # 使用字典类型作为通用的状态模式
                # 在较新版本的langgraph中，可以直接使用dict作为状态模式
                self.map = StateGraph(dict)
                logger.info("成功创建空map，使用dict作为状态模式")
                return True
        except Exception as e:
            logger.error(f"设置map时出错: {e}")
            return False
    
    def check_map(self):
        """
        检查并输出map图结构
        
        Returns:
            dict or None: map的结构信息，如果map不存在则返回None
        """
        if self.map is None:
            logger.warning("map不存在")
            return None
        
        try:
            # 获取map的结构信息
            map_structure = {}
            
            # 尝试获取节点信息
            if hasattr(self.map, 'nodes'):
                map_structure['nodes'] = list(self.map.nodes)
            elif hasattr(self.map, 'get_nodes'):
                map_structure['nodes'] = self.map.get_nodes()
            else:
                map_structure['nodes'] = "无法获取节点信息"
            
            # 尝试获取边信息
            if hasattr(self.map, 'edges'):
                map_structure['edges'] = list(self.map.edges)
            elif hasattr(self.map, 'get_edges'):
                map_structure['edges'] = self.map.get_edges()
            else:
                map_structure['edges'] = "无法获取边信息"
            
            # 尝试获取其他可能的属性
            if hasattr(self.map, 'metadata'):
                map_structure['metadata'] = self.map.metadata
            
            logger.info(f"map结构信息: {map_structure}")
            return map_structure
        except Exception as e:
            logger.error(f"检查map结构时出错: {e}")
            return None
    
    def add_node(self, node_name: str, node_func):
        """
        向map中添加节点
        
        Args:
            node_name (str): 节点名称
            node_func: 节点函数，用于处理输入状态并返回更新后的状态
            
        Returns:
            bool: 添加是否成功
        """
        try:
            # 检查map是否存在
            if self.map is None:
                logger.error("map不存在，请先创建map")
                return False
            
            # 检查map是否有add_node方法
            if not hasattr(self.map, 'add_node'):
                logger.error("map对象不支持添加节点操作")
                return False
            
            # 添加节点
            self.map.add_node(node_name, node_func)
            logger.info(f"成功添加节点: {node_name}")
            return True
        except Exception as e:
            logger.error(f"添加节点时出错: {e}")
            return False
    
    def compile_map(self):
        """
        检查map属性是否存在，如果存在则进行编译
        
        Returns:
            object: 编译后的图对象(app)，如果编译成功
            None: 如果map不存在或编译失败
        """
        logger = logging.getLogger(__name__)
        logger.info("开始编译map...")
        
        try:
            # 检查map属性是否存在
            if not hasattr(self, 'map') or self.map is None:
                logger.warning("map属性不存在或为None，无法编译")
                return None
            
            # 检查map是否有compile方法
            if not hasattr(self.map, 'compile'):
                logger.warning("map对象没有compile方法")
                return None
            
            # 执行编译操作
            logger.info("正在编译map...")
            compiled_app = self.map.compile()
            logger.info("map编译成功")
            
            # 保存编译结果（可选）
            self.compiled_app = compiled_app  # 可以保存编译结果以备后用
            
            return compiled_app
            
        except Exception as e:
            logger.error(f"map编译失败: {str(e)}")
            return None