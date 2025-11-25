# 在文件顶部的导入部分添加typing导入
import os
import logging
import sys
from typing import Dict, Any, List  # 确保类型提示始终可用

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from tool.config_load import load_config_to_env
from langchain_community.vectorstores import FAISS
# 替换HuggingFaceEmbeddings为ZhipuAiClient
from zai import ZhipuAiClient
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
#from langchain_core.tools import BaseTool, StructuredTool, Tool, tool
# 为langgraph图节点和agent工具添加必要的导入（仅导入langgraph相关模块）
try:
    from langchain_core.tools import BaseTool, StructuredTool, Tool, tool
    LANGGRAPH_AVAILABLE = True
except ImportError:
    logger.warning("langgraph或langchain工具相关模块未安装，某些功能将不可用")
    LANGGRAPH_AVAILABLE = False



# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 创建一个自定义的Embeddings类，适配LangChain的接口
class ZhipuAIEmbeddings(Embeddings):
    def __init__(self, api_key, model="embedding-3"):
        self.client = ZhipuAiClient(api_key=api_key)
        self.model = model
    
    def embed_documents(self, texts):
        """为文档列表生成嵌入向量"""
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=texts
            )
            # 假设response中包含embeddings字段，每个元素有embedding属性
            return [item.embedding for item in response.data]
        except Exception as e:
            logger.error(f"生成文档嵌入失败: {str(e)}")
            raise
    
    def embed_query(self, text):
        """为单个查询生成嵌入向量"""
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=[text]
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"生成查询嵌入失败: {str(e)}")
            raise

class RAG:
    def __init__(self):
        """
        初始化RAG类，加载配置并设置默认参数
        """
        logger.info("初始化RAG类...")
        
        # 加载配置
        try:
            load_config_to_env()
            logger.info("配置加载成功")
        except Exception as e:
            logger.error(f"配置加载失败: {str(e)}")
            raise
        
        # 保存向量库的默认地址和相关信息
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.vectorstore_path = os.environ.get('VECTORSTORE_PATH', os.path.join(base_dir, 'LLM_base', 'RAG', 'vectorstore'))
        self.documents_path = os.path.join(base_dir, 'LLM_base', 'RAG', 'documents')
        # 获取智谱AI API密钥
        self.api_key = os.environ.get('API_KEY')
        if not self.api_key:
            raise ValueError("未找到ZHIPU_API_KEY环境变量，请配置智谱AI API密钥")
        self.embedding_model = "embedding-3"
        self.vectorstore_info = {
            'path': self.vectorstore_path,
            'embedding_model': self.embedding_model,
            'type': 'FAISS',
            'documents_path': self.documents_path
        }
        logger.info(f"向量库默认配置: {self.vectorstore_info}")
        
        # 检查并创建文档目录
        try:
            if not os.path.exists(self.documents_path):
                os.makedirs(self.documents_path, exist_ok=True)
                logger.info(f"创建文档目录: {self.documents_path}")
            else:
                logger.info(f"文档目录已存在: {self.documents_path}")
        except Exception as e:
            logger.error(f"创建文档目录失败: {str(e)}")
            raise
        
        # 检查并创建向量库目录
        try:
            if not os.path.exists(self.vectorstore_path):
                os.makedirs(self.vectorstore_path, exist_ok=True)
                logger.info(f"创建向量库目录: {self.vectorstore_path}")
            else:
                logger.info(f"向量库目录已存在: {self.vectorstore_path}")
        except Exception as e:
            logger.error(f"创建向量库目录失败: {str(e)}")
            raise
        
        # 空的缓存检索句柄类属性
        self.vectorstore = None
        self.embeddings = None
        
        logger.info("RAG类初始化完成")

    # 其他方法保持不变，只修改create_vectorstore和get_vectorstore中初始化embeddings的部分
    def create_vectorstore(self, vectorstore_path=None, embedding_model=None):
        """
        先检查向量库是否存在，没有则，使用指定方法，在指定地址创建向量库
        
        Args:
            vectorstore_path (str, optional): 向量库路径，如果为None则使用默认路径
            embedding_model (str, optional): 嵌入模型，如果为None则使用默认模型
            
        Returns:
            bool: 创建是否成功
        """
        path = vectorstore_path or self.vectorstore_path
        model = embedding_model or self.embedding_model
        logger.info(f"准备创建向量库，路径: {path}, 嵌入模型: {model}")
        
        # 先检查向量库是否存在
        if self.check_vectorstore_exists(path):
            logger.info(f"向量库已存在，跳过创建: {path}")
            return True
        
        try:
            # 确保目录存在
            os.makedirs(path, exist_ok=True)
            logger.info(f"创建向量库目录: {path}")
            
            # 初始化智谱AI嵌入模型
            logger.info(f"初始化智谱AI嵌入模型: {model}")
            self.embeddings = ZhipuAIEmbeddings(api_key=self.api_key, model=model)
            
            # 创建空的向量库（需要至少一个文档）
            # 这里创建一个示例文档作为占位符
            dummy_doc = Document(page_content="向量库初始化文档", metadata={"source": "dummy"})
            
            # 创建向量库
            logger.info("创建空向量库...")
            vectorstore = FAISS.from_documents([dummy_doc], self.embeddings)
            
            # 保存向量库
            vectorstore.save_local(path)
            logger.info(f"向量库创建成功并保存到: {path}")
            
            # 更新向量库信息
            self.vectorstore_info.update({
                'path': path,
                'embedding_model': model,
                'created': True
            })
            
            return True
            
        except Exception as e:
            logger.error(f"创建向量库失败: {str(e)}")
            return False
    
    def get_vectorstore(self):
        """
        检查句柄类属性，如果没有，则检查向量库，有加载向量库，将句柄放入类属性中，并返回
        
        Returns:
            FAISS or None: 向量库实例，如果加载失败则返回None
        """
        # 检查句柄是否已存在
        if self.vectorstore is not None:
            logger.info("向量库句柄已存在，直接返回")
            return self.vectorstore
        
        # 检查向量库是否存在
        if not self.check_vectorstore_exists():
            logger.error("向量库不存在，无法加载")
            return None
        
        try:
            # 初始化智谱AI嵌入模型
            if self.embeddings is None:
                logger.info(f"初始化智谱AI嵌入模型: {self.embedding_model}")
                self.embeddings = ZhipuAIEmbeddings(api_key=self.api_key, model=self.embedding_model)
            
            # 加载向量库
            logger.info(f"加载向量库: {self.vectorstore_path}")
            self.vectorstore = FAISS.load_local(
                self.vectorstore_path,
                self.embeddings,
                allow_dangerous_deserialization=True  # 注意：生产环境中应谨慎使用
            )
            
            logger.info("向量库加载成功")
            return self.vectorstore
            
        except Exception as e:
            logger.error(f"加载向量库失败: {str(e)}")
            return None

# 其余方法保持不变

    def check_vectorstore_exists(self, vectorstore_path=None):
        """
        检查指定位置的向量库是否存在，并输出向量库相关信息
        
        Args:
            vectorstore_path (str, optional): 向量库路径，如果为None则使用默认路径
            
        Returns:
            bool: 向量库是否存在
        """
        path = vectorstore_path or self.vectorstore_path
        logger.info(f"检查向量库是否存在，路径: {path}")
        
        try:
            # 检查向量库文件是否存在
            if not os.path.exists(path):
                logger.warning(f"向量库路径不存在: {path}")
                return False
            
            # 检查必要的索引文件
            index_files = ['index.faiss', 'index.pkl']
            all_files_exist = all(os.path.exists(os.path.join(path, file)) for file in index_files)
            
            if all_files_exist:
                logger.info(f"向量库存在，路径: {path}")
                logger.info(f"向量库信息: {self.vectorstore_info}")
                return True
            else:
                missing_files = [file for file in index_files if not os.path.exists(os.path.join(path, file))]
                logger.warning(f"向量库路径存在但缺少必要文件: {missing_files}")
                return False
                
        except Exception as e:
            logger.error(f"检查向量库失败: {str(e)}")
            return False
    
    def create_vectorstore(self, vectorstore_path=None, embedding_model=None):
        """
        先检查向量库是否存在，没有则，使用指定方法，在指定地址创建向量库
        
        Args:
            vectorstore_path (str, optional): 向量库路径，如果为None则使用默认路径
            embedding_model (str, optional): 嵌入模型，如果为None则使用默认模型
            
        Returns:
            bool: 创建是否成功
        """
        path = vectorstore_path or self.vectorstore_path
        model = embedding_model or self.embedding_model
        logger.info(f"准备创建向量库，路径: {path}, 嵌入模型: {model}")
        
        # 先检查向量库是否存在
        if self.check_vectorstore_exists(path):
            logger.info(f"向量库已存在，跳过创建: {path}")
            return True
        
        try:
            # 确保目录存在
            os.makedirs(path, exist_ok=True)
            logger.info(f"创建向量库目录: {path}")
            
            # 初始化嵌入模型
            logger.info(f"初始化嵌入模型: {model}")
            self.embeddings = ZhipuAIEmbeddings(api_key=self.api_key, model=model)
            
            # 创建空的向量库（需要至少一个文档）
            # 这里创建一个示例文档作为占位符
            dummy_doc = Document(page_content="向量库初始化文档", metadata={"source": "dummy"})
            
            # 创建向量库
            logger.info("创建空向量库...")
            vectorstore = FAISS.from_documents([dummy_doc], self.embeddings)
            
            # 保存向量库
            vectorstore.save_local(path)
            logger.info(f"向量库创建成功并保存到: {path}")
            
            # 更新向量库信息
            self.vectorstore_info.update({
                'path': path,
                'embedding_model': model,
                'created': True
            })
            
            return True
            
        except Exception as e:
            logger.error(f"创建向量库失败: {str(e)}")
            return False
    
    def get_vectorstore(self):
        """
        检查句柄类属性，如果没有，则检查向量库，有加载向量库，将句柄放入类属性中，并返回
        
        Returns:
            FAISS or None: 向量库实例，如果加载失败则返回None
        """
        # 检查句柄是否已存在
        if self.vectorstore is not None:
            logger.info("向量库句柄已存在，直接返回")
            return self.vectorstore
        
        # 检查向量库是否存在
        if not self.check_vectorstore_exists():
            logger.error("向量库不存在，无法加载")
            return None
        
        try:
            # 初始化嵌入模型
            if self.embeddings is None:
                logger.info(f"初始化嵌入模型: {self.embedding_model}")
                self.embeddings = ZhipuAIEmbeddings(api_key=self.api_key, model=self.embedding_model)
            
            # 加载向量库
            logger.info(f"加载向量库: {self.vectorstore_path}")
            self.vectorstore = FAISS.load_local(
                self.vectorstore_path,
                self.embeddings,
                allow_dangerous_deserialization=True  # 注意：生产环境中应谨慎使用
            )
            
            logger.info("向量库加载成功")
            return self.vectorstore
            
        except Exception as e:
            logger.error(f"加载向量库失败: {str(e)}")
            return None
    
    def add_document_to_vectorstore(self, file_path):
        """
        先检查向量库是否存在，存在则将指定文件加入向量库
        
        Args:
            file_path (str): 要添加的文件路径
            
        Returns:
            bool: 添加是否成功
        """
        logger.info(f"准备添加文件到向量库: {file_path}")
        
        # 检查文件是否存在
        if not os.path.exists(file_path):
            logger.error(f"要添加的文件不存在: {file_path}")
            return False
        
        # 检查向量库是否存在
        if not self.check_vectorstore_exists():
            logger.error("向量库不存在，请先创建向量库")
            return False
        
        try:
            # 获取或加载向量库
            vectorstore = self.get_vectorstore()
            if vectorstore is None:
                logger.error("无法获取向量库实例")
                return False
            
            # 根据文件类型选择合适的加载器
            file_ext = os.path.splitext(file_path)[1].lower()
            if file_ext == '.txt':
                loader = TextLoader(file_path, encoding='utf-8')
            elif file_ext == '.pdf':
                loader = PyPDFLoader(file_path)
            elif file_ext in ['.docx', '.doc']:
                loader = Docx2txtLoader(file_path)
            else:
                logger.error(f"不支持的文件类型: {file_ext}")
                return False
            
            # 加载文档
            logger.info(f"加载文档: {file_path}")
            documents = loader.load()
            
            # 分割文档
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50
            )
            logger.info("分割文档...")
            split_docs = text_splitter.split_documents(documents)
            logger.info(f"文档分割完成，共{len(split_docs)}个chunk")
            
            # 添加到向量库
            logger.info("将文档添加到向量库...")
            vectorstore.add_documents(split_docs)
            
            # 保存更新后的向量库
            vectorstore.save_local(self.vectorstore_path)
            logger.info(f"文件成功添加到向量库并保存: {file_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"添加文件到向量库失败: {str(e)}")
            return False
    def search_knowledge_base(self, query, k=3):
        """
        检索知识库，返回与查询相关的文档和来源文件
        
        Args:
            query (str): 查询文本
            k (int, optional): 返回的最相关文档数量，默认为3
            
        Returns:
            dict: 包含检索结果和来源文件的字典
                {"results": [检索到的文档列表], "sources": [来源文件列表]}
                如果检索失败返回None
        """
        logger.info(f"开始检索知识库，查询: '{query}'，返回前{k}个结果")
        
        try:
            # 检查是否有句柄，如果没有则获取
            if self.vectorstore is None:
                logger.info("向量库句柄不存在，尝试获取...")
                self.vectorstore = self.get_vectorstore()
                
                if self.vectorstore is None:
                    logger.error("无法获取向量库句柄，检索失败")
                    return None
            
            # 执行相似性搜索
            logger.info("执行相似性搜索...")
            docs_with_scores = self.vectorstore.similarity_search_with_score(query, k=k)
            
            # 提取结果和来源文件信息
            results = []
            sources = set()
            
            for doc, score in docs_with_scores:
                # 构建结果项
                result_item = {
                    "content": doc.page_content,
                    "score": float(score),  # 转换为float以便序列化
                    "metadata": doc.metadata
                }
                results.append(result_item)
                
                # 提取来源文件信息
                if "source" in doc.metadata:
                    sources.add(doc.metadata["source"])
            
            logger.info(f"检索完成，找到{len(results)}个相关文档，来自{len(sources)}个不同来源")
            
            # 返回结果
            return {
                "results": results,
                "sources": list(sources)
            }
            
        except Exception as e:
            logger.error(f"检索知识库失败: {str(e)}")
            return None

# 添加新函数：将RAG包装成langgraph的图节点
def rag_node(state: Dict[str, Any], rag_instance: RAG = None, k: int = 3) -> Dict[str, Any]:
    """
    将RAG包装成langgraph的图节点
    
    Args:
        state: 包含查询的状态字典，需要包含'query'键
        rag_instance: 可选的RAG实例，如果为None则创建新实例
        k: 返回的最相关文档数量，默认为3
    
    Returns:
        更新后的状态字典，添加'retrieved_context'键，包含检索结果
    """
    if not LANGGRAPH_AVAILABLE:
        logger.error("langgraph模块不可用，请安装langgraph")
        return state
    
    try:
        # 确保RAG实例存在
        if rag_instance is None:
            rag_instance = RAG()
        
        # 从状态中获取查询
        query = state.get('query', '')
        print(state)
        if not query:
            logger.error("状态字典中未找到'query'键")
            return state
        
        # 执行知识库检索
        logger.info(f"RAG节点执行检索，查询: '{query}'")
        search_results = rag_instance.search_knowledge_base(query, k=k)
        
        if search_results:
            # 将检索结果添加到状态中
            state['retrieved_context'] = search_results
            logger.info(f"RAG节点检索完成，找到{len(search_results['results'])}个相关文档")
        else:
            state['retrieved_context'] = None
            logger.warning("RAG节点检索失败，未返回任何结果")
        
        return state
        
    except Exception as e:
        logger.error(f"RAG节点执行失败: {str(e)}")
        state['retrieved_context'] = None
        state['error'] = str(e)
        return state

# 添加新函数：将RAG包装成agent的tool
def create_rag_tool(rag_instance: RAG = None, name: str = "search_knowledge_base", 
                   description: str = "搜索知识库以获取相关信息") -> StructuredTool:
    """
    将RAG包装成agent的tool
    
    Args:
        rag_instance: 可选的RAG实例，如果为None则创建新实例
        name: 工具名称
        description: 工具描述
    
    Returns:
        结构化的工具对象，可以添加到agent中
    """
    if not LANGGRAPH_AVAILABLE:
        logger.error("langchain工具相关模块不可用，请安装相关依赖")
        return None
    
    try:
        # 确保RAG实例存在
        if rag_instance is None:
            rag_instance = RAG()
        
        # 定义工具函数
        def rag_tool(query: str, k: int = 3) -> str:
            """
            搜索知识库获取相关信息
            
            Args:
                query: 要搜索的查询文本
                k: 返回的最相关文档数量，默认为3
            
            Returns:
                搜索结果的文本摘要
            """
            results = rag_instance.search_knowledge_base(query, k=k)
            
            if not results or not results['results']:
                return "未找到相关信息"
            
            # 构建响应文本
            response_text = f"找到{len(results['results'])}个相关信息：\n\n"
            for i, result in enumerate(results['results'], 1):
                content = result['content'][:200] + ("..." if len(result['content']) > 200 else "")
                response_text += f"[{i}] 相似度: {result['score']:.4f}\n内容: {content}\n\n"
            
            if results['sources']:
                response_text += f"来源文件: {', '.join(results['sources'])}"
            
            return response_text
        
        # 创建结构化工具
        tool = StructuredTool.from_function(
            func=rag_tool,
            name=name,
            description=description,
            args_schema=None  # 使用自动推导的参数模式
        )
        
        logger.info(f"成功创建RAG工具: {name}")
        return tool
        
    except Exception as e:
        logger.error(f"创建RAG工具失败: {str(e)}")
        return None

# 示例使用代码
if __name__ == "__main__":
    try:
        # 创建RAG实例
        rag = RAG()
        
        # 检查向量库是否存在
        exists = rag.check_vectorstore_exists()
        print(f"向量库存在: {exists}")
        
        # 如果不存在则创建
        if not exists:
            created = rag.create_vectorstore()
            print(f"向量库创建: {created}")
        
        # 加载向量库
        vectorstore = rag.get_vectorstore()
        print(f"向量库加载: {vectorstore is not None}")
        
        # 示例：添加文件（需要替换为实际文件路径）
        # add_success = rag.add_document_to_vectorstore("example.txt")
        # print(f"文件添加: {add_success}")
        
    except Exception as e:
        print(f"错误: {str(e)}")