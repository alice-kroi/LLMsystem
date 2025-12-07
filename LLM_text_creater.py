import os
import sys
import logging
import uuid
from typing import Dict, Any, Optional

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入必要的模块
from LLM_base.Agent import Agent, create_agent_node
from LLM_base.map import LLMMap
from LLM_base.prompt import PromptLoader
from LLM_base.RAG import RAG

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NovelWritingSystem:
    def __init__(self, config_path=r'e:\GitHub\config.yaml'):
        """
        初始化小说编写系统
        
        Args:
            config_path (str, optional): 配置文件路径
        """
        self.config_path = config_path
        self.agents = {}
        self.rag = None
        self.graph = None
        self.prompt_loader = None
        
        # 初始化系统组件
        self._initialize_components()
        
        # 创建多Agent协作图
        self._create_agent_graph()
    
    def _initialize_components(self):
        """
        初始化系统组件
        """
        # 初始化提示词加载器
        self.prompt_loader = PromptLoader()
        
        # 初始化RAG系统用于知识库检索
        try:
            self.rag = RAG()
            logger.info("RAG系统初始化成功")
        except Exception as e:
            logger.warning(f"RAG系统初始化失败: {e}，将在需要时使用通用知识")
    
    def _create_agent_graph(self):
        """
        创建多Agent协作图
        """
        try:
            # 创建LLMMap实例
            llm_map = LLMMap(config_path=self.config_path)
            
            # 创建状态图
            llm_map.set_map()
            
            # 使用create_agent_node函数创建Agent节点
            def create_process_node(agent_role: str, prompt_type: str, next_step_key: str):
                """
                创建处理节点
                
                Args:
                    agent_role: 代理角色名称
                    prompt_type: 提示词类型
                    next_step_key: 下一步数据的键名
                    
                Returns:
                    节点处理函数
                """
                # 创建专用Agent节点
                agent_node = create_agent_node(config_path=self.config_path)
                
                def process_node(state: Dict[str, Any]) -> Dict[str, Any]:
                    """处理节点函数"""
                    try:
                        # 根据不同的节点角色构建输入提示
                        if agent_role == 'outline_creator':
                            # 大纲创建节点 - 使用主题和知识库信息
                            topic = state.get('topic')
                            relevant_info = self._retrieve_from_knowledge_base(topic)
                            search_query = f"{topic}\n\n参考信息: {relevant_info}"
                            prompt = self.prompt_loader.get_formatted_prompt(prompt_type, search_query)
                        elif agent_role == 'character_developer':
                            # 角色塑造节点 - 使用大纲
                            outline = state.get('outline')
                            if not outline:
                                return {**state, 'error': '缺少大纲信息', 'status': 'error'}
                            prompt = self.prompt_loader.get_formatted_prompt(prompt_type, outline)
                        elif agent_role == 'plot_developer':
                            # 情节发展节点 - 组合大纲和角色信息
                            outline = state.get('outline')
                            character = state.get('character')
                            if not outline or not character:
                                return {**state, 'error': '缺少大纲或角色信息', 'status': 'error'}
                            combined_info = f"大纲:\n{outline}\n\n角色信息:\n{character}"
                            prompt = self.prompt_loader.get_formatted_prompt(prompt_type, combined_info)
                        elif agent_role == 'emotional_writer':
                            # 情感描写节点 - 组合情节和角色信息
                            plot = state.get('plot')
                            character = state.get('character')
                            if not plot or not character:
                                return {**state, 'error': '缺少情节或角色信息', 'status': 'error'}
                            combined_info = f"情节:\n{plot}\n\n角色信息:\n{character}"
                            prompt = self.prompt_loader.get_formatted_prompt(prompt_type, combined_info)
                        elif agent_role == 'story_polisher':
                            # 故事润色节点 - 使用情感内容
                            emotional_content = state.get('emotional_content')
                            if not emotional_content:
                                return {**state, 'error': '缺少情感内容信息', 'status': 'error'}
                            prompt = self.prompt_loader.get_formatted_prompt(prompt_type, emotional_content)
                        else:
                            return {**state, 'error': f'未知的代理角色: {agent_role}', 'status': 'error'}
                        
                        # 准备节点输入状态
                        node_input = {
                            'prompt': prompt,
                            'conversation_id': state.get('conversation_id')
                        }
                        
                        # 调用Agent节点处理
                        result = agent_node(node_input)
                        
                        # 检查结果并更新状态
                        if result.get('status') == 'success':
                            updated_state = {
                                **state,
                                next_step_key: result.get('response'),
                                'conversation_id': result.get('conversation_id'),
                                'status': f'{next_step_key}_created'
                            }
                            logger.info(f"{agent_role}节点成功处理并生成{next_step_key}")
                            return updated_state
                        else:
                            error_msg = result.get('error', f'{agent_role}处理失败')
                            logger.error(f"{agent_role}节点处理失败: {error_msg}")
                            return {**state, 'error': error_msg, 'status': 'error'}
                    except Exception as e:
                        logger.error(f"{agent_role}节点执行时出错: {e}")
                        return {**state, 'error': str(e), 'status': 'error'}
                
                return process_node
            
            # 创建各个处理节点
            outline_node = create_process_node('outline_creator', 'story_outline_creator', 'outline')
            character_node = create_process_node('character_developer', 'character_developer', 'character')
            plot_node = create_process_node('plot_developer', 'plot_developer', 'plot')
            emotional_node = create_process_node('emotional_writer', 'emotional_writer', 'emotional_content')
            polish_node = create_process_node('story_polisher', 'story_polisher', 'final_story')
            
            # 添加节点到图中
            llm_map.add_node('outline_creator', outline_node)
            llm_map.add_node('character_developer', character_node)
            llm_map.add_node('plot_developer', plot_node)
            llm_map.add_node('emotional_writer', emotional_node)
            llm_map.add_node('story_polisher', polish_node)
            
            # 设置边（线性流程）
            if hasattr(llm_map.map, 'add_edge'):
                llm_map.map.add_edge('outline_creator', 'character_developer')
                llm_map.map.add_edge('character_developer', 'plot_developer')
                llm_map.map.add_edge('plot_developer', 'emotional_writer')
                llm_map.map.add_edge('emotional_writer', 'story_polisher')
            
            # 设置入口点
            if hasattr(llm_map.map, 'set_entry_point'):
                llm_map.map.set_entry_point('outline_creator')
            
            # 设置出口点
            if hasattr(llm_map.map, 'set_finish_point'):
                llm_map.map.set_finish_point('story_polisher')
            
            # 编译图
            self.graph = llm_map.compile_map()
            logger.info("多Agent协作图创建成功")
        except Exception as e:
            logger.error(f"创建Agent协作图失败: {e}")
    
    def _retrieve_from_knowledge_base(self, query: str) -> str:
        """
        从知识库检索相关信息
        
        Args:
            query: 查询内容
            
        Returns:
            str: 检索到的相关信息
        """
        if not self.rag:
            return "知识库暂时不可用，将基于通用知识进行创作"
        
        try:
            # 这里简化处理，实际应该调用RAG的检索方法
            # 假设RAG类有一个retrieve方法
            if hasattr(self.rag, 'retrieve'):
                results = self.rag.retrieve(query, k=3)
                return "\n\n".join([f"相关资料 {i+1}: {r}" for i, r in enumerate(results)])
            else:
                logger.warning("RAG实例没有retrieve方法")
                return "知识库检索功能暂不可用"
        except Exception as e:
            logger.error(f"知识库检索失败: {e}")
            return "知识库检索失败，将基于通用知识进行创作"
    
    def write_novel(self, topic: str) -> Dict[str, Any]:
        """
        开始编写小说
        
        Args:
            topic: 小说主题
            
        Returns:
            Dict: 包含创作过程和最终故事的字典
        """
        if not self.graph:
            logger.error("Agent协作图未初始化成功")
            return {'error': '系统初始化失败', 'status': 'error'}
        
        try:
            # 创建会话ID
            conversation_id = str(uuid.uuid4())
            logger.info(f"开始创作小说，主题: {topic}, 会话ID: {conversation_id}")
            
            # 运行图
            initial_state = {
                'topic': topic,
                'conversation_id': conversation_id,
                'status': 'started'
            }
            
            result = self.graph.invoke(initial_state)
            
            # 检查最终状态
            if 'final_story' in result and result.get('final_story'):
                result['status'] = 'story_completed'
                logger.info("小说创作成功完成")
                # 保存最终故事
                self._save_story(result)
            else:
                logger.warning(f"小说创作未完全成功，状态: {result.get('status', 'unknown')}")
            
            return result
        except Exception as e:
            logger.error(f"小说创作过程中出错: {e}")
            return {'error': str(e), 'status': 'error'}
    
    def _save_story(self, result: Dict[str, Any]):
        """
        保存创作的故事
        
        Args:
            result: 包含故事内容的结果字典
        """
        try:
            # 创建保存目录
            stories_dir = os.path.join(os.path.dirname(__file__), 'stories')
            os.makedirs(stories_dir, exist_ok=True)
            
            # 创建文件名
            topic = result.get('topic', 'untitled').replace(' ', '_')[:50]
            filename = f"{topic}_{result.get('conversation_id', 'unknown')}.txt"
            filepath = os.path.join(stories_dir, filename)
            
            # 写入文件
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"主题: {result.get('topic', '未知')}\n\n")
                f.write(f"【故事大纲】\n{result.get('outline', '未生成')}\n\n")
                f.write(f"【角色设定】\n{result.get('character', '未生成')}\n\n")
                f.write(f"【详细情节】\n{result.get('plot', '未生成')}\n\n")
                f.write(f"【情感描写】\n{result.get('emotional_content', '未生成')}\n\n")
                f.write(f"【最终故事】\n{result.get('final_story', '未生成')}\n\n")
            
            logger.info(f"故事已保存到: {filepath}")
        except Exception as e:
            logger.error(f"保存故事失败: {e}")

def main():
    """
    主函数
    """
    print("===== 女性视角自我挑战小说创作系统 =====")
    print("本系统将通过多个AI角色协作，创作以女性第一视角展开的自我挑战故事")
    
    try:
        # 初始化系统
        print("正在初始化创作系统...")
        system = NovelWritingSystem()
        print("系统初始化完成！")
        
        while True:
            print("\n请输入你想创作的小说主题（输入'退出'结束程序）:")
            topic = input("主题: ").strip()
            
            if topic.lower() in ['退出', 'exit', 'quit']:
                print("感谢使用小说创作系统，再见！")
                break
            
            if not topic:
                print("主题不能为空，请重新输入")
                continue
            
            print(f"\n开始创作主题为 '{topic}' 的小说...")
            print("正在进行故事大纲创作...")
            
            # 开始创作
            result = system.write_novel(topic)
            
            if result.get('status') == 'story_completed':
                print("\n🎉 小说创作成功完成！")
                print("\n【最终故事预览】")
                # 显示故事开头部分
                final_story = result.get('final_story', '')
                preview = final_story[:500] + "..." if len(final_story) > 500 else final_story
                print(preview)
                print("\n完整故事已保存到系统中")
            else:
                print(f"\n创作过程中遇到问题: {result.get('error', '未知错误')}")
                print("请检查系统配置或稍后重试")
                
    except KeyboardInterrupt:
        print("\n\n程序已被用户中断")
    except Exception as e:
        print(f"\n系统运行出错: {e}")
        print("请检查系统环境和配置")

if __name__ == "__main__":
    main()