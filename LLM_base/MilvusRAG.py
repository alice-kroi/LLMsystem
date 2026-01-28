import os
import logging
import sys
import time
import uuid
from typing import List, Dict, Any
import numpy as np
from openai import OpenAI
from langchain_core.embeddings import Embeddings
from tool.config_load import load_config_to_env
load_config_to_env()
# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 导入新版MilvusClient
from pymilvus import MilvusClient

# 定义DoubaoEmbeddings类
class DoubaoEmbeddings(Embeddings):
    def __init__(self, api_key=None, model="doubao-embedding-text-240715"):
        print("初始化豆包嵌入模型")
        if not api_key:
            api_key = os.getenv("Doubao_API_KEY")
        print(api_key)
        # 如果没有提供api_key，可以使用环境变量或默认值
        self.client = OpenAI(
            base_url="https://ark.cn-beijing.volces.com/api/v3",
            api_key=api_key # 可以替换为环境变量
        )
        self.model = model
        self.vector_dim = 768  # 明确向量维度
    
    def embed_documents(self, texts):
        """为文档列表生成嵌入向量"""
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=texts,
                dimensions=self.vector_dim
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            logger.error(f"豆包生成文档嵌入失败: {str(e)}")
            # 如果API调用失败，返回随机向量作为备选
            return [np.random.random(self.vector_dim).tolist() for _ in texts]
    
    def embed_query(self, text):
        """为单个查询生成嵌入向量"""
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=[text],
                dimensions=self.vector_dim
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"豆包生成查询嵌入失败: {str(e)}")
            # 如果API调用失败，返回随机向量作为备选
            return np.random.random(self.vector_dim).tolist()

class MilvusRAG:
    def __init__(self, uri="http://localhost:19530", token="root:Milvus", dbname="vtuber", embedding_model=DoubaoEmbeddings()):
        """
        初始化Milvus RAG类（使用新版MilvusClient API）
        
        Args:
            uri (str): Milvus服务地址，默认为http://localhost:19530
            token (str): 连接令牌，默认为root:Milvus
            dbname (str): 数据库名称，默认为vtuber
            embedding_model: 嵌入模型实例，用于生成文本向量
        """
        logger.info("初始化MilvusRAG类...")
        
        try:
            # 使用新版MilvusClient连接到Milvus服务
            self.client = MilvusClient(
                uri=uri,
                token=token
            )
            logger.info(f"成功连接到Milvus服务: {uri}")
            
            # 设置当前数据库为vtuber
            self.client.use_database(db_name=dbname)
            logger.info(f"成功切换到数据库: {dbname}")
            
            # 定义聊天历史集合名称
            self.chat_history_collection_name = "chat_history"
            
            # 嵌入模型
            self.embedding_model = embedding_model
            
            # 创建或加载集合
            self._create_or_load_collection()
            
            logger.info("MilvusRAG类初始化完成")
            
        except Exception as e:
            logger.error(f"初始化MilvusRAG失败: {str(e)}")
            raise
    
    def _create_or_load_collection(self):
        """
        创建或加载聊天历史集合（使用新版MilvusClient API）
        """
        # 检查集合是否存在
        if not self.client.has_collection(collection_name=self.chat_history_collection_name):
            # 定义集合字段（新版API格式）
            fields = [
                {
                    "name": "message_id",
                    "data_type": "VARCHAR",
                    "max_length": 36,
                    "is_primary": True,
                    "description": "消息ID"
                },
                {
                    "name": "user_id",
                    "data_type": "VARCHAR",
                    "max_length": 100,
                    "description": "用户ID"
                },
                {
                    "name": "username",
                    "data_type": "VARCHAR",
                    "max_length": 100,
                    "description": "用户名"
                },
                {
                    "name": "content",
                    "data_type": "VARCHAR",
                    "max_length": 10000,
                    "description": "消息内容"
                },
                {
                    "name": "timestamp",
                    "data_type": "DOUBLE",
                    "description": "时间戳"
                },
                {
                    "name": "message_type",
                    "data_type": "VARCHAR",
                    "max_length": 10,
                    "description": "消息类型：query或response"
                },
                {
                    "name": "vector_field",
                    "data_type": "FLOAT_VECTOR",
                    "dim": 768,
                    "description": "消息向量"
                }
            ]
            
            # 创建集合（修复：直接传递fields参数，不使用schema字典）
            self.client.create_collection(
                collection_name=self.chat_history_collection_name,
                description="聊天历史集合",
                fields=fields
            )
            logger.info(f"创建聊天历史集合: {self.chat_history_collection_name}")
        else:
            logger.info(f"加载聊天历史集合: {self.chat_history_collection_name}")
    
    def _generate_vector(self, content: str) -> List[float]:
        """
        使用DoubaoEmbeddings生成向量
        
        Args:
            content (str): 内容文本
            
        Returns:
            List[float]: 生成的向量
        """
        return self.embedding_model.embed_query(content)
    
    def construct_data_input(self, user_id: str, username: str, content: str, message_type: str) -> Dict[str, Any]:
        """
        构造数据输入
        
        Args:
            user_id (str): 用户ID
            username (str): 用户名
            content (str): 消息内容
            message_type (str): 消息类型（query或response）
            
        Returns:
            Dict[str, Any]: 构造的数据字典
        """
        message_id = str(uuid.uuid4())
        timestamp = int(time.time())
        
        # 生成向量
        vector = self._generate_vector(content)
        
        data = {
            "message_id": message_id,
            "user_id": user_id,
            "username": username,
            "content": content,
            "timestamp": timestamp,
            "message_type": message_type,
            "vector_field": vector
        }
        
        logger.info(f"构造数据输入完成: {message_id}")
        return data
    
    def add_message(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        添加消息到Milvus
        
        Args:
            data (Dict[str, Any]): 消息数据
            
        Returns:
            Dict[str, Any]: 添加的消息信息
        """
        # 插入数据（新版API格式）
        result = self.client.insert(
            collection_name=self.chat_history_collection_name,
            data=[data]
        )
        
        logger.info(f"添加消息成功: {data['message_id']}")
        return data
    
    def add_user_message(self, user_id: str, username: str, content: str) -> Dict[str, Any]:
        """
        添加用户消息到Milvus
        
        Args:
            user_id (str): 用户ID
            username (str): 用户名
            content (str): 消息内容
            
        Returns:
            Dict[str, Any]: 添加的消息信息
        """
        data = self.construct_data_input(user_id, username, content, "query")
        return self.add_message(data)
    
    def add_llm_response(self, user_id: str, username: str, content: str) -> Dict[str, Any]:
        """
        添加LLM回复到Milvus
        
        Args:
            user_id (str): 用户ID
            username (str): 用户名
            content (str): 回复内容
            
        Returns:
            Dict[str, Any]: 添加的回复信息
        """
        data = self.construct_data_input(user_id, username, content, "response")
        return self.add_message(data)
    
    def semantic_similarity_search(self, query: str, top_k: int = 5, user_id: str = None) -> List[Dict[str, Any]]:
        """
        语义相似度查询
        
        Args:
            query (str): 查询文本
            top_k (int): 返回结果数量，默认为5
            user_id (str, optional): 用户ID，用于筛选特定用户的消息
            
        Returns:
            List[Dict[str, Any]]: 相似的消息列表
        """
        logger.info(f"执行语义相似度查询: {query}，返回前{top_k}个结果")
        
        # 生成查询向量
        query_vector = self._generate_vector(query)
        
        # 构建过滤条件
        filter_expr = f"user_id == '{user_id}'" if user_id else ""
        
        # 执行向量搜索
        results = self.client.search(
            collection_name=self.chat_history_collection_name,
            data=[query_vector],
            limit=top_k,
            output_fields=["message_id", "user_id", "username", "content", "timestamp", "message_type"],
            search_params={"metric_type": "COSINE", "params": {}},
            filter=filter_expr
        )
        
        # 处理结果
        similar_messages = []
        for result in results[0]:
            similar_messages.append({
                "message_id": result["entity"]["message_id"],
                "user_id": result["entity"]["user_id"],
                "username": result["entity"]["username"],
                "content": result["entity"]["content"],
                "timestamp": result["entity"]["timestamp"],
                "message_type": result["entity"]["message_type"],
                "distance": result["distance"],
                "similarity": 1 - result["distance"]  # 余弦距离转换为相似度
            })
        
        logger.info(f"语义相似度查询完成，共返回{len(similar_messages)}条记录")
        return similar_messages
    
    def search_by_username(self, username: str, limit: int = 20, offset: int = 0) -> List[Dict[str, Any]]:
        """
        用户名称查询
        
        Args:
            username (str): 用户名
            limit (int): 返回结果数量，默认为20
            offset (int): 结果偏移量，默认为0
            
        Returns:
            List[Dict[str, Any]]: 查询到的消息列表
        """
        logger.info(f"按用户名查询: {username}，限制{limit}条，偏移{offset}")
        
        # 构建查询条件
        query_expr = f"username == '{username}'"
        
        # 执行查询，按时间戳排序
        results = self.client.query(
            collection_name=self.chat_history_collection_name,
            filter=query_expr,
            limit=limit,
            offset=offset,
            output_fields=["message_id", "user_id", "username", "content", "timestamp", "message_type"],
            consistency_level="Strong",
            order_by="timestamp DESC"
        )
        
        logger.info(f"按用户名查询完成，共返回{len(results)}条记录")
        return results
    
    def search_recent_questions(self, user_id: str = None, limit: int = 20) -> List[Dict[str, Any]]:
        """
        最近问题查询
        
        Args:
            user_id (str, optional): 用户ID，用于筛选特定用户的问题
            limit (int): 返回结果数量，默认为20
            
        Returns:
            List[Dict[str, Any]]: 最近的问题列表
        """
        logger.info(f"查询最近问题，用户ID: {user_id}，限制{limit}条")
        
        # 构建查询条件
        if user_id:
            query_expr = f"user_id == '{user_id}' && message_type == 'query'"
        else:
            query_expr = "message_type == 'query'"
        
        # 执行查询，按时间戳倒序排序
        results = self.client.query(
            collection_name=self.chat_history_collection_name,
            filter=query_expr,
            limit=limit,
            output_fields=["message_id", "user_id", "username", "content", "timestamp"],
            consistency_level="Strong",
            order_by="timestamp DESC"
        )
        
        logger.info(f"最近问题查询完成，共返回{len(results)}条记录")
        return results
    
    def get_chat_history(self, user_id: str, limit: int = 20, offset: int = 0) -> List[Dict[str, Any]]:
        """
        获取指定用户的聊天历史
        
        Args:
            user_id (str): 用户ID
            limit (int): 返回结果的数量限制，默认为20
            offset (int): 结果偏移量，默认为0
            
        Returns:
            List[Dict[str, Any]]: 聊天历史记录列表
        """
        logger.info(f"获取用户{user_id}的聊天历史，限制{limit}条，偏移{offset}")
        
        # 构建查询条件
        query_expr = f"user_id == '{user_id}'"
        
        # 执行查询，按时间戳排序
        results = self.client.query(
            collection_name=self.chat_history_collection_name,
            filter=query_expr,
            limit=limit,
            offset=offset,
            output_fields=["message_id", "user_id", "username", "content", "timestamp", "message_type"],
            consistency_level="Strong",
            order_by="timestamp DESC"
        )
        
        logger.info(f"获取聊天历史成功，共返回{len(results)}条记录")
        return results
    
    def get_chat_history_by_time_range(self, user_id: str, start_time: float, end_time: float) -> List[Dict[str, Any]]:
        """
        获取指定用户在时间范围内的聊天历史
        
        Args:
            user_id (str): 用户ID
            start_time (float): 开始时间戳（Unix时间）
            end_time (float): 结束时间戳（Unix时间）
            
        Returns:
            List[Dict[str, Any]]: 聊天历史记录列表
        """
        logger.info(f"获取用户{user_id}在时间范围[{start_time}, {end_time}]内的聊天历史")
        
        # 构建查询条件
        query_expr = f"user_id == '{user_id}' && timestamp >= {start_time} && timestamp <= {end_time}"
        
        # 执行查询，按时间戳排序
        results = self.client.query(
            collection_name=self.chat_history_collection_name,
            filter=query_expr,
            output_fields=["message_id", "user_id", "username", "content", "timestamp", "message_type"],
            consistency_level="Strong",
            order_by="timestamp ASC"
        )
        
        logger.info(f"获取时间范围聊天历史成功，共返回{len(results)}条记录")
        return results
    
    def delete_message(self, message_id: str) -> bool:
        """
        删除指定的聊天消息
        
        Args:
            message_id (str): 消息ID
            
        Returns:
            bool: 删除成功返回True，否则返回False
        """
        logger.info(f"删除消息ID为{message_id}的聊天记录")
        
        # 执行删除操作
        result = self.client.delete(
            collection_name=self.chat_history_collection_name,
            filter=f"message_id == '{message_id}'"
        )
        
        logger.info(f"删除消息成功，影响行数: {result['deleted_count']}")
        return result['deleted_count'] > 0
    
    def delete_chat_history(self, user_id: str) -> bool:
        """
        删除指定用户的所有聊天历史
        
        Args:
            user_id (str): 用户ID
            
        Returns:
            bool: 删除成功返回True，否则返回False
        """
        logger.info(f"删除用户{user_id}的所有聊天历史")
        
        # 执行删除操作
        result = self.client.delete(
            collection_name=self.chat_history_collection_name,
            filter=f"user_id == '{user_id}'"
        )
        #print(result)
        
        logger.info(f"删除用户聊天历史成功，影响行数: {result['delete_count']}")
        return result['delete_count'] > 0
    
    def count_messages(self, user_id: str) -> int:
        """
        统计指定用户的消息数量
        
        Args:
            user_id (str): 用户ID
            
        Returns:
            int: 消息数量
        """
        logger.info(f"统计用户{user_id}的消息数量")
        
        # 执行统计
        count = self.client.query(
            collection_name=self.chat_history_collection_name,
            filter=f"user_id == '{user_id}'",
            output_fields=["count(*)"]
        )
        
        # 提取统计结果
        msg_count = count[0]["count(*)"] if count else 0
        logger.info(f"用户{user_id}共有{msg_count}条消息")
        return msg_count
    
    def close(self):
        """
        关闭Milvus连接（新版API不需要显式关闭连接）
        """
        logger.info("Milvus连接已关闭（新版API自动管理连接）")

# 示例使用代码
if __name__ == "__main__":
    try:
        # 初始化MilvusRAG实例
        rag = MilvusRAG(
            uri="http://localhost:19530",
            token="root:Milvus",
            dbname="vtuber"
        )
        
        # 测试添加用户消息和LLM回复
        user_id = "test_user_001"
        username = "测试用户"
        
        # 添加用户消息
        user_msg = rag.add_user_message(user_id, username, "你好，我是用户")
        print(f"添加用户消息: {user_msg}")
        
        # 添加LLM回复
        llm_resp = rag.add_llm_response(user_id, username, "你好，我是AI助手，有什么可以帮助你的吗？")
        print(f"添加LLM回复: {llm_resp}")
        
        # 测试构造数据输入
        constructed_data = rag.construct_data_input(user_id, username, "测试构造数据输入", "query")
        print(f"\n构造数据输入: {constructed_data}")
        
        # 测试语义相似度查询
        similar_messages = rag.semantic_similarity_search("你好", top_k=2)
        print(f"\n语义相似度查询结果:")
        for msg in similar_messages:
            print(f"  内容: {msg['content']}，相似度: {msg['similarity']:.4f}")
        
        # 测试用户名称查询
        user_messages = rag.search_by_username("测试用户", limit=5)
        print(f"\n用户名称查询结果:")
        for msg in user_messages:
            print(f"  {msg['message_type'].upper()}: {msg['content']}")
        
        # 测试最近问题查询
        recent_questions = rag.search_recent_questions(user_id, limit=5)
        print(f"\n最近问题查询结果:")
        for msg in recent_questions:
            print(f"  {msg['content']} (时间: {msg['timestamp']})")
        
        # 获取聊天历史
        history = rag.get_chat_history(user_id, limit=10)
        print(f"\n聊天历史:")
        for msg in history:
            print(f"{msg['message_type'].upper()}: {msg['content']} (时间: {msg['timestamp']})")
        
        # 获取消息数量
        msg_count = rag.count_messages(user_id)
        print(f"\n用户{user_id}共有{msg_count}条消息")
        
        # 关闭连接
        rag.close()
        print("\n程序运行完成！")
        
    except Exception as e:
        logger.error(f"程序运行出错: {str(e)}")
        import traceback
        traceback.print_exc()

# 完整的主测试函数
if __name__ == "__main__":
    try:
        # 导入必要的库
        import uuid
        
        # 初始化MilvusRAG实例
        print("=== 初始化MilvusRAG ===")
        rag = MilvusRAG(
            uri="http://localhost:19530",
            token="root:Milvus",
            dbname="vtuber"
        )
        
        # 准备测试数据
        test_user_id = str(uuid.uuid4())
        test_username = "测试用户"
        
        print(f"\n测试用户ID: {test_user_id}")
        print(f"测试用户名: {test_username}")
        
        # 测试1: 构造数据输入
        print("\n=== 测试1: 构造数据输入 ===")
        test_content = "我想了解Milvus向量数据库的基本概念"
        constructed_data = rag.construct_data_input(
            user_id=test_user_id,
            username=test_username,
            content=test_content,
            message_type="query"
        )
        print(f"构造的数据: {constructed_data}")
        
        # 将构造的数据添加到Milvus
        added_message = rag.add_message(constructed_data)
        print(f"添加到Milvus成功: {added_message['message_id']}")
        
        # 添加更多测试数据
        test_messages = [
            {"content": "Milvus支持哪些向量索引类型？", "type": "query"},
            {"content": "向量数据库的主要应用场景有哪些？", "type": "query"},
            {"content": "如何在Python中使用Milvus？", "type": "query"},
            {"content": "Milvus是一个高性能、可扩展的向量数据库，专为AI应用设计。", "type": "response"}
        ]
        
        for msg in test_messages:
            rag.add_message(rag.construct_data_input(
                user_id=test_user_id,
                username=test_username,
                content=msg["content"],
                message_type=msg["type"]
            ))
        
        print("\n已添加5条测试消息到Milvus")
        
        # 测试2: 语义相似度查询
        print("\n=== 测试2: 语义相似度查询 ===")
        query_text = "Milvus的基本概念是什么？"
        similar_results = rag.semantic_similarity_search(
            query=query_text,
            top_k=3
        )
        
        print(f"查询文本: {query_text}")
        print(f"找到 {len(similar_results)} 条相似消息:")
        for i, result in enumerate(similar_results, 1):
            print(f"{i}. 内容: {result['content']}")
            print(f"   类型: {result['message_type']}")
            print(f"   相似度: {result['similarity']:.4f}")
            print(f"   时间: {result['timestamp']}")
            print()
        
        # 测试3: 用户名称查询
        print("=== 测试3: 用户名称查询 ===")
        username_results = rag.search_by_username(
            username=test_username,
            limit=5
        )
        
        print(f"用户名: {test_username}")
        print(f"找到 {len(username_results)} 条消息:")
        for i, result in enumerate(username_results, 1):
            print(f"{i}. 类型: {result['message_type']}")
            print(f"   内容: {result['content']}")
            print(f"   时间: {result['timestamp']}")
            print()
        
        # 测试4: 最近问题查询
        print("=== 测试4: 最近问题查询 ===")
        recent_questions = rag.search_recent_questions(
            user_id=test_user_id,
            limit=3
        )
        
        print(f"用户 {test_username} 的最近问题:")
        for i, result in enumerate(recent_questions, 1):
            print(f"{i}. 内容: {result['content']}")
            print(f"   时间: {result['timestamp']}")
            print()
        
        # 测试5: 获取聊天历史
        print("=== 测试5: 获取聊天历史 ===")
        chat_history = rag.get_chat_history(
            user_id=test_user_id,
            limit=10
        )
        
        print(f"用户 {test_username} 的聊天历史:")
        for i, msg in enumerate(chat_history, 1):
            print(f"{i}. {msg['message_type'].upper()}: {msg['content']}")
            print(f"   时间: {msg['timestamp']}")
            print()
        
        # 测试6: 统计消息数量
        print("=== 测试6: 统计消息数量 ===")
        msg_count = rag.count_messages(test_user_id)
        print(f"用户 {test_username} 共有 {msg_count} 条消息")
        
        # 测试7: 删除测试数据（可选）
        print("=== 测试7: 清理测试数据 ===")
        delete_result = rag.delete_chat_history(test_user_id)
        print(f"删除用户 {test_username} 的所有消息: {'成功' if delete_result else '失败'}")
        
        # 关闭连接
        rag.close()
        print("\n=== 所有测试完成 ===")
        
    except Exception as e:
        print(f"测试过程中出错: {str(e)}")
        import traceback
        traceback.print_exc()