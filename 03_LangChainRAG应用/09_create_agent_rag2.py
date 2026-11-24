# ============================================================
# 原生 pymilvus 建库写入 + langchain_milvus 封装层挂载检索的 Agentic RAG
# ============================================================
from langchain.agents import create_agent
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_milvus import Milvus
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import create_retriever_tool
from pymilvus import MilvusClient, DataType

from env_utils import DASHSCOPE_API_KEY
from init_llm import deepseek_llm_flash

MILVUS_URI = "http://localhost:19530"
MILVUS_DB_NAME = "default"
COLLECTION_NAME = "rag_course_native"

# ============================================================
# 第一步：初始化真实 Embedding 模型（建库和检索共用同一个）
# ============================================================
embedding_model = DashScopeEmbeddings(
    model="text-embedding-v1",
    dashscope_api_key=DASHSCOPE_API_KEY,
)


# ============================================================
# 第二步：用 pymilvus 原生方式建表 + 建索引 + 写入
# ============================================================
def build_collection_natively(client: MilvusClient, dimension: int) -> None:
    """原生创建Collection，字段名必须与后面挂载封装层时的参数严格一致。"""
    if client.has_collection(COLLECTION_NAME):
        client.drop_collection(COLLECTION_NAME)

    # 开启动态字段，source 可作为动态字段写入并自动回到 Document.metadata
    schema = client.create_schema(auto_id=True, enable_dynamic_field=True)
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True, auto_id=True)
    schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=1000)
    schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=dimension)

    # 建立索引
    index_params = client.prepare_index_params()

    index_params.add_index(
        field_name="vector",
        index_type="AUTOINDEX",
        metric_type="COSINE",
    )

    client.create_collection(
        collection_name=COLLECTION_NAME,
        schema=schema,
        index_params=index_params,
    )


def insert_documents(client: MilvusClient) -> None:
    """用原生 insert 写入售后规则。"""
    rows = [
        {"text": "退款规则：仓库验收后发起原路退款，通常三个工作日内到账。", "source": "退款规则.md"},
        {"text": "物流规则：订单发货后可以在订单详情查看物流轨迹。", "source": "物流规则.md"},
    ]

    vectors = embedding_model.embed_documents([row["text"] for row in rows])

    client.insert(
        collection_name=COLLECTION_NAME,
        data=[{**row, "vector": vector} for row, vector in zip(rows, vectors)]
    )

    # 刷新 Collection 状态，确保数据写入完成
    client.flush(collection_name=COLLECTION_NAME)

    # 加载 Collection，确保索引生效
    client.load_collection(collection_name=COLLECTION_NAME)


milvus_client = MilvusClient(uri=MILVUS_URI, db_name=MILVUS_DB_NAME)
dimension = len(embedding_model.embed_query("测试文本"))
build_collection_natively(milvus_client, dimension=dimension)
insert_documents(milvus_client)

# ============================================================
# 第三步：用 LangChain 封装层挂载同一个已存在的 Collection
# ============================================================
vector_store = Milvus(
    embedding_function=embedding_model, # 用于将查询文本转换为向量
    collection_name=COLLECTION_NAME, # 指定要挂载的集合名
    connection_args={"uri": MILVUS_URI, "db_name": MILVUS_DB_NAME}, # 连接 Milvus 的参数
    auto_id=True,  # 主键由 Milvus 自动生成
    primary_field="id", # 主键字段名
    text_field="text", # 文本字段名
    vector_field="vector", # 向量字段名
    enable_dynamic_field=True,  # 允许动态字段

    # index_params：告诉 LangChain 本集合用的是哪种度量，用于把原始余弦距离换算成相关性分数
    index_params={
        "index_type": "AUTOINDEX",
        "metric_type": "COSINE",
        "params": {},
    },
    # search_params：发给 Milvus 服务端的检索参数，决定用哪种度量做近邻搜索
    search_params={
        "metric_type": "COSINE",
        "params": {}
    }
)

# ============================================================
# 第四步：把 Retriever 包装成 Agent 工具
# ============================================================
document_prompt = PromptTemplate.from_template("{page_content}\n来源：{source}")

retriever_tool = create_retriever_tool(
    retriever=vector_store.as_retriever(search_kwargs={"k": 1}),
    name="rag_course_native_retriever",
    description="检索退款到账和物流规则。遇到退款、到账或物流问题必须调用。",
    document_prompt=document_prompt,
)

if __name__ == '__main__':
    # ========================================================
    # 第五步：使用 create_agent 创建 Agent
    # ========================================================
    agent = create_agent(
        model=deepseek_llm_flash,
        tools=[retriever_tool],
        system_prompt=(
            "你是电商售后 Agent。涉及退款、到账或物流问题时必须先调用检索工具。只能根据工具结果回答，回答末尾列出来源。"),
    )

    # ========================================================
    # 第六步：调用 Agent 并观察消息循环
    # ========================================================
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "退款多久到账?"}]}
    )
    for msg in result["messages"]:
        msg.pretty_print()

    print("*" * 100)
    print("\n最终回答：", result["messages"][-1].content)
