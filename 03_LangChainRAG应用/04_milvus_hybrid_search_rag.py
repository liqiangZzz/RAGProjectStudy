# ==============================================================================
# 使用 Milvus 原生 BM25 + 向量检索 + RRF 构建混合检索 RAG
# 流程：BM25(精确词项) + 向量检索(语义召回) -> RRF融合 -> 组装Prompt -> DeepSeek回答
# ==============================================================================
from typing import List, Dict, Any, Tuple

from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage
from pymilvus import MilvusClient, DataType, Function, FunctionType, AnnSearchRequest, RRFRanker

from env_utils import DASHSCOPE_API_KEY
from init_llm import deepseek_llm_flash

# ============================================================
# 全局配置
# ============================================================
MILVUS_URI = "http://localhost:19530"
MILVUS_DB_NAME = "default"
COLLECTION_NAME = "tmp_course_hybrid_search"


# ============================================================
# 第一步：Collection 创建与初始化
# ============================================================
def init_collection(client: MilvusClient, embedding_dimension: int) -> None:
    """创建同时支持向量检索和BM25检索的Collection。"""
    if client.has_collection(COLLECTION_NAME):
        client.drop_collection(COLLECTION_NAME)

    # auto_id=True: 自动生成主键; enable_dynamic_field=False: 禁止未定义字段插入
    schema = client.create_schema(auto_id=True, enable_dynamic_field=False)

    # 1. 主键（Milvus自动生成）
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True, auto_id=True)

    # 2. 业务文档唯一标识
    schema.add_field(field_name="doc_id", datatype=DataType.VARCHAR, max_length=100)

    # 3. 文档正文（同时作为BM25的输入字段）
    schema.add_field(
        field_name="text",
        datatype=DataType.VARCHAR,
        max_length=1000,
        enable_analyzer=True,  # 开启分词器
        analyzer_params={"tokenizer": "jieba"},  # 中文使用jieba分词
    )
    # 4. 元数据字段（用于展示和过滤）
    schema.add_field(field_name="source", datatype=DataType.VARCHAR, max_length=200)  # 文档来源
    schema.add_field(field_name="product", datatype=DataType.VARCHAR, max_length=100)  # 产品类型，用于过滤
    schema.add_field(field_name="tenant_id", datatype=DataType.VARCHAR, max_length=100)  # 租户标识，用于权限隔离

    # 5. 稠密向量字段（用于语义检索）
    schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=embedding_dimension)

    # 6. 稀疏向量字段（用于BM25关键词检索）
    schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)

    # 定义Milvus内置的BM25函数：自动将text字段转换为sparse_vector稀疏向量
    schema.add_function(
        Function(
            name="bm25_function",
            function_type=FunctionType.BM25,
            input_field_names=["text"],  # 输入字段，BM25需要文档正文
            output_field_names=["sparse_vector"],  # 输出字段，BM25稀疏向量
        )
    )

    # 定义索引参数
    index_params = client.prepare_index_params()

    # 稠密向量索引：AUTOINDEX自动选择最优索引，COSINE余弦相似度
    index_params.add_index(
        field_name="dense_vector",
        index_type="AUTOINDEX",
        metric_type="COSINE",
    )

    # 稀疏向量索引：SPARSE_INVERTED_INDEX倒排索引，BM25算法计算相关性
    index_params.add_index(
        field_name="sparse_vector",
        index_type="SPARSE_INVERTED_INDEX",
        metric_type="BM25",
    )

    # 创建Collection
    client.create_collection(
        collection_name=COLLECTION_NAME,
        index_params=index_params,
        schema=schema,
    )

# ============================================================
# 第二步：数据准备与插入
# ============================================================
def get_mock_documents() -> List[Dict[str, Any]]:
    """准备跨产品、跨租户的售后测试文档。"""
    return [
        {"doc_id": "model-ab120", "text": "AB-120 是降噪耳机型号。", "source": "产品型号.md", "product": "降噪耳机", "tenant_id": "tenant_a"},
        {"doc_id": "refund-noise", "text": "降噪耳机退货后，退款通常三个工作日内到账。", "source": "退款规则.md", "product": "降噪耳机", "tenant_id": "tenant_a"},
        {"doc_id": "model-ab121", "text": "AB-121 是运动耳机型号。", "source": "产品型号.md", "product": "运动耳机", "tenant_id": "tenant_a"},
        {"doc_id": "price-tenant-b", "text": "AB-120 的内部批发价格是保密信息。", "source": "内部价格.md", "product": "降噪耳机", "tenant_id": "tenant_b"},
    ]


def insert_documents(
        client: MilvusClient,
        embedding_model: DashScopeEmbeddings,
        documents: List[Dict[str, Any]]
) -> None:
    """为文档生成稠密向量并写入 Milvus"""
    # 1. 批量生成稠密向量
    dense_vectors = embedding_model.embed_documents([doc["text"] for doc in documents])

    # 2. 组装写入数据 (sparse_vector由Milvus自动生成，无需手动传入)
    insert_data = [
        {**doc, "dense_vector": vector}
        for doc, vector in zip(documents, dense_vectors)
    ]

    # 3. 插入并确保落盘
    client.insert(collection_name=COLLECTION_NAME, data=insert_data)
    client.flush(collection_name=COLLECTION_NAME)  # 确保数据写入磁盘
    client.load_collection(collection_name=COLLECTION_NAME)  # 搜索前必须 load


# ============================================================
# 第三步：RAG 检索与生成
# ============================================================
def format_context(results: list[dict]) -> str:
    """将Milvus检索结果转换成带来源的上下文。"""
    if not results:
        return "无相关参考资料。"

    return "\n\n".join(
        f"[{rank}] 来源：{hit['entity']['source']}\n{hit['entity']['text']}"
        for rank, hit in enumerate(results, start=1)
    )


def retrieve_hybrid_context(
        client: MilvusClient,
        embedding_model: DashScopeEmbeddings,
        query: str,
        filter_expr: str
) -> Tuple[List[dict], str]:
    """执行混合检索，返回原始结果和格式化后的上下文"""

    # 1. 将用户问题转为稠密向量
    query_dense_vector = embedding_model.embed_query(query)

    # 2. 构造稠密向量语义检索请求
    dense_request = AnnSearchRequest(
        data=[query_dense_vector],
        anns_field="dense_vector",
        param={"metric_type": "COSINE", "params": {}},
        limit=3,
        expr=filter_expr,  # 强制应用元数据过滤
    )

    # 3. 构造 BM25 关键词检索请求
    sparse_request = AnnSearchRequest(
        data=[query],  # 直接传原始文本，Milvus内部自动分词转换
        anns_field="sparse_vector",
        param={"metric_type": "BM25", "params": {}},
        limit=3,
        expr=filter_expr,  # 强制应用元数据过滤
    )

    # 4. 执行混合检索 (两路召回，RRF融合)
    hybrid_results = client.hybrid_search(
        collection_name=COLLECTION_NAME,
        reqs=[dense_request, sparse_request],
        ranker=RRFRanker(k=60),  # k=60 为RRF平滑常数
        limit=2,  # 最终取Top2
        output_fields=["doc_id", "text", "source", "product", "tenant_id"],
    )[0]  # 取出单条查询的结果

    # 5. 格式化为大模型需要的上下文
    context = format_context(hybrid_results)
    return hybrid_results, context


def generate_answer(query: str, context: str) -> str:
    """调用 DeepSeek 模型生成最终回答"""
    messages = [
        SystemMessage(content=(
            "你是售后助手。只能根据参考资料回答，必须保留时间和条件，最后列出来源。"
        )),
        HumanMessage(content=(
            f"参考资料：\n{context}\n\n用户问题：{query}"
        ))
    ]

    try:
        response = deepseek_llm_flash.invoke(messages)
        return response.content
    except Exception as e:
        return f"大模型调用失败: {e}"


# ============================================================
# 主流程
# ============================================================
if __name__ == '__main__':
    # 1. 初始化客户端与模型
    client = MilvusClient(uri=MILVUS_URI, db_name=MILVUS_DB_NAME)
    embedding_model = DashScopeEmbeddings(
        model="text-embedding-v1",
        dashscope_api_key=DASHSCOPE_API_KEY
    )

    try:
        # 2. 获取数据并建表
        documents = get_mock_documents()
        # 先 embed 一条获取维度
        temp_dim = len(embedding_model.embed_query("测试维度"))
        init_collection(client, temp_dim)

        # 3. 插入数据并加载
        insert_documents(client, embedding_model, documents)

        # 4. 执行 RAG 检索
        question = "AB-120 退货后多久到账？"
        # 过滤条件：只查产品为“降噪耳机”且租户为“tenant_a”的数据（保证数据隔离）
        filter_expression = 'product == "降噪耳机" and tenant_id == "tenant_a"'

        hybrid_results, context = retrieve_hybrid_context(client, embedding_model, question, filter_expression)

        # 5. 打印检索中间状态（方便调试）
        print(f"\n=== 查询问题 ===\n{question}")
        print(f"=== 过滤条件 ===\n{filter_expression}")
        print("=== 检索融合结果 ===")
        for rank, hit in enumerate(hybrid_results, start=1):
            print(f"{rank}. rrf_score= {hit['distance']:.6f} | doc_id= {hit['entity']['doc_id']}")
            print(f"   内容：{hit['entity']['text']}")
            print(f"   来源：{hit['entity']['source']}")

        # 6. 生成最终回答
        print("\n=== 正在调用大模型生成回答 ===")
        answer = generate_answer(question, context)
        print(f"最终回答：\n{answer}")

    finally:
        pass
        # 安全清理：无论代码是否报错，最后都销毁临时Collection
        # 如果你想保留数据用于调试，可以注释掉下面两行
        # if client.has_collection(COLLECTION_NAME):
        #     client.drop_collection(COLLECTION_NAME)
        #     print("\n[清理] 临时 Collection 已删除。")
