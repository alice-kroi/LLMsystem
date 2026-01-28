from pymilvus import MilvusClient, DataType
from openai import OpenAI
from langchain_core.embeddings import Embeddings
import logging
import uuid
import time
import numpy as np

logger = logging.getLogger(__name__)

class DoubaoEmbeddings(Embeddings):
    def __init__(self, api_key=None, model="doubao-embedding-text-240715"):
        # 如果没有提供api_key，可以使用环境变量或默认值
        self.client = OpenAI(
            base_url="https://ark.cn-beijing.volces.com/api/v3",
            api_key=api_key or "cf1d0e35-d99b-4189-8c69-92a175619833"  # 可以替换为环境变量
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

# 初始化Milvus客户端
client = MilvusClient(
    uri="http://localhost:19530",
    token="root:Milvus",
    db_name="vtuber"
)

# 检查并创建数据库
try:
    client.create_database(db_name="vtuber")
    print("成功创建数据库: vtuber")
except Exception as e:
    print(f"数据库已存在或创建失败: {e}")

# 获取数据库信息
describe = client.describe_database(db_name="vtuber")
print("数据库信息:", describe)

# 创建集合 - 聊天历史集合
collection_name = "chat_history"

# 删除已存在的集合（测试用）
if client.has_collection(collection_name):
    client.drop_collection(collection_name)
    print(f"已删除现有集合: {collection_name}")

# 正确的方式：使用schema创建collection
# 1. 先创建schema
schema = client.create_schema(
    auto_id=False,  # 不使用自动ID，使用自定义UUID
    enable_dynamic_field=True  # 允许动态字段
)

# 2. 逐个添加字段 - 使用DataType常量
schema.add_field(
    field_name="message_id",
    datatype=DataType.VARCHAR,  # 使用DataType常量
    max_length=36,
    is_primary=True
)

schema.add_field(
    field_name="user_id",
    datatype=DataType.VARCHAR,  # 使用DataType常量
    max_length=36
)

schema.add_field(
    field_name="username",
    datatype=DataType.VARCHAR,  # 使用DataType常量
    max_length=50
)

schema.add_field(
    field_name="content",
    datatype=DataType.VARCHAR,  # 使用DataType常量
    max_length=2000
)

schema.add_field(
    field_name="timestamp",
    datatype=DataType.INT64  # 使用DataType常量
)

schema.add_field(
    field_name="message_type",
    datatype=DataType.VARCHAR,  # 使用DataType常量
    max_length=20
)

schema.add_field(
    field_name="vector_field",
    datatype=DataType.FLOAT_VECTOR,  # 使用DataType常量
    dim=2560
)

# 3. 使用schema创建collection
index_params=client.prepare_index_params()
index_params.add_index(
    field_name="vector_field", # Name of the scalar field to be indexed
    index_type="", # Type of index to be created. For auto indexing, leave it empty or omit this parameter.
    index_name="default_index" # Name of the index to be created
)
client.create_collection(
    collection_name=collection_name,
    schema=schema,
    consistency_level="Strong",  # 设置强一致性\
)
client.create_index(
    collection_name=collection_name,
    index_params=index_params
)


#print(f"成功创建集合: {collection_name}")

# 查看集合信息
collection_info = client.describe_collection(collection_name)
#print("集合信息:", collection_info)

# 初始化嵌入模型
embedding_model = DoubaoEmbeddings()


res = client.list_collections()

print(res)
# 7. Load the collection
client.load_collection(
    collection_name=collection_name
)

res = client.get_load_state(
    collection_name=collection_name
)

print(res)
# 插入测试数据
def insert_test_data():
    print("\n=== 插入测试数据 ===")
    # 生成3条测试消息
    test_messages = [
        {"user_id": str(uuid.uuid4()), "username": "测试用户1", "content": "你好，我想了解一下Milvus", "message_type": "user_message"},
        {"user_id": str(uuid.uuid4()), "username": "测试用户1", "content": "Milvus支持哪些向量索引类型？", "message_type": "user_message"},
        {"user_id": str(uuid.uuid4()), "username": "测试用户2", "content": "向量数据库的应用场景有哪些？", "message_type": "user_message"}
    ]
    
    # 为每条消息生成嵌入向量
    contents = [msg["content"] for msg in test_messages]
    vectors = embedding_model.embed_documents(contents)
    print(len(vectors[0]))
    # 准备插入数据
    insert_data = []
    for i, msg in enumerate(test_messages):
        insert_data.append({
            "message_id": str(uuid.uuid4()),
            "user_id": msg["user_id"],
            "username": msg["username"],
            "content": msg["content"],
            "timestamp": int(time.time()),
            "message_type": msg["message_type"],
            "vector_field": vectors[i]
        })
    
    # 插入数据
    result = client.insert(
        collection_name=collection_name,
        data=insert_data
    )
    print(f"成功插入 {len(insert_data)} 条数据")
    return insert_data

# 查询测试数据
# 查询测试数据
def query_test_data(query_text, top_k=2):
    print(f"\n=== 查询测试: '{query_text}' ===")
    # 生成查询向量
    query_vector = embedding_model.embed_query(query_text)
    
    # 执行向量搜索
    results = client.search(
        collection_name=collection_name,
        data=[query_vector],  # 搜索数据
        anns_field="vector_field",  # 向量字段名
        limit=top_k,  # 返回结果数量
        output_fields=["message_id", "user_id", "username", "content", "timestamp", "message_type"],  # 返回的字段
        metric_type="COSINE",  # 相似度度量方式
        params={"nprobe": 10}  # 搜索参数
    )
    
    # 显示查询结果
    for i, hits in enumerate(results):
        print(f"查询结果 {i+1}:")
        for j, hit in enumerate(hits):
            print(f"  结果 {j+1}:")
            print(f"    消息ID: {hit['message_id']}")
            print(f"    用户: {hit['username']}")
            print(f"    内容: {hit['content']}")
            print(f"    时间戳: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(hit['timestamp']))}")
            print(f"    消息类型: {hit['message_type']}")
            print(f"    相似度: {hit['distance']:.4f}")
    
    return results
# 标量查询测试
def scalar_query_test(username):
    print(f"\n=== 标量查询测试: 查询用户 '{username}' 的所有消息 ===")
    results = client.query(
        collection_name=collection_name,
        filter=f"username == '{username}'",  # 查询条件
        output_fields=["message_id", "user_id", "username", "content", "timestamp", "message_type"]  # 返回的字段
    )
    
    print(f"找到 {len(results)} 条消息:")
    for msg in results:
        print(f"  消息ID: {msg['message_id']}")
        print(f"    内容: {msg['content']}")
        print(f"    时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(msg['timestamp']))}")
    
    return results

# 统计消息数量
# 使用正确的查询方法统计消息数量
def count_messages():
    print("\n=== 统计消息数量 ===")
    # 使用client.query查询所有消息，然后计算数量
    results = client.query(
        collection_name=collection_name,
        filter="message_id IS NOT NULL",  # 条件始终为真，查询所有记录
        output_fields=["message_id"],  # 只返回message_id字段，减少数据传输
        limit=1000  # 设置一个合理的限制
    )
    
    message_count = len(results)
    print(f"总消息数量: {message_count}")
    return message_count

# 分页查询示例

# 主测试函数
def run_tests():
    # 插入测试数据
    insert_test_data()
    
    # 统计消息数量
    count_messages()
    
    # 向量查询测试
    query_test_data("Milvus是什么？", top_k=2)
    query_test_data("向量索引有哪些？", top_k=2)
    
    # 标量查询测试
    scalar_query_test("测试用户1")


# 运行测试
if __name__ == "__main__":
    

    run_tests()