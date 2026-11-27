#  Milvus Hybrid Search 混合检索
#  同一个问题同时执行向量检索和BM25检索，再使用RRF融合结果
from typing import List

from langchain_community.embeddings import DashScopeEmbeddings
from pymilvus import MilvusClient, DataType, Function, FunctionType, AnnSearchRequest, WeightedRanker

from env_utils import DASHSCOPE_API_KEY

# 创建千问Embedding模型
embedding_model = DashScopeEmbeddings(
    model="text-embedding-v1",  # 指定Embedding模型名称
    dashscope_api_key=DASHSCOPE_API_KEY,  # 设置DashScope API Key
)

MILVUS_URI = "http://localhost:19530"  # Milvus服务地址
MILVUS_DB_NAME = "default"  # Milvus数据库名称
COLLECTION_NAME = "tmp_simple_hybrid_search"


def create_collection(client: MilvusClient):
    """创建包含文本字段和BM25稀疏向量的集合"""

    # 如果Collection已存在，先删除
    if client.has_collection(collection_name=COLLECTION_NAME):
        client.drop_collection(collection_name=COLLECTION_NAME)

    # ====================================================
    # 第一步：定义Collection字段
    # ====================================================
    # 创建Collection Schema
    schema = client.create_schema(
        # 主键（id 字段）由 Milvus 自动生成（递增整数）
        auto=True,
        # 不允许动态添加字段，插入数据时只能使用 schema 中定义的字段，不能额外添加未定义的字段。
        enable_dynamic_field=False
    )

    # 获取当前 Embedding 模型的向量维度
    dimension = len(embedding_model.embed_query("测试文本"))

    # 主键字段有 Milvus 自动生成，递增整数
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True, auto_id=True)

    # BM25 的输入的文本字段
    schema.add_field(
        field_name="text",
        datatype=DataType.VARCHAR,
        max_length=1024,
        # 当这个字段的数据要转成向量（或者倒排索引）时，请用 jieba 把它切碎
        enable_analyzer=True,  # 开启文本分析功能，BM25需要使用
        analyzer_params={"tokenizer": "jieba"},  # 中文使用 jieba 分词
    )

    # 文档来源
    schema.add_field(field_name="source", datatype=DataType.VARCHAR, max_length=200)

    #  添加稠密向量字段
    schema.add_field(
        field_name="dense_vector",  # 稠密向量字段
        datatype=DataType.FLOAT_VECTOR,  # 浮点数向量类型
        dim=dimension,  # 使用真实Embedding维度
    )

    # 添加BM25稀疏向量字段
    schema.add_field(
        field_name="sparse_vector",  # 保存BM25产生的稀疏向量
        datatype=DataType.SPARSE_FLOAT_VECTOR,  # 使用稀疏向量类型
    )

    # ====================================================
    # 第二步：定义Milvus原生BM25函数
    # ====================================================
    # Milvus 原生内置 BM25 算法，自动将 text 字段转换为 sparse_vector
    schema.add_function(
        Function(  # 创建函数定义
            name="bm25_vector",  # 自定义函数名称
            function_type=FunctionType.BM25,  # 指定为 BM25 转换函数
            input_field_names=["text"],  # 输入字段为 text
            output_field_names=["sparse_vector"],  # 输出字段为 sparse_vector
        )
    )

    # ---------------- 定义索引参数 ----------------
    index_params = client.prepare_index_params()

    # 稠密向量索引（使用 AUTOINDEX 自动选择最优索引，使用 COSINE 计算余弦相似度）
    index_params.add_index(
        field_name="dense_vector",  # 指定稠密向量字段
        index_type="AUTOINDEX",  # 由Milvus自动选择索引配置
        metric_type="COSINE",  # 使用余弦相似度比较语义
    )
    # 稀疏向量索引（使用倒排索引 SPARSE_INVERTED_INDEX，使用 BM25 算法计算相关性）
    index_params.add_index(
        field_name="sparse_vector",  # 索引字段为 sparse_vector
        index_type="SPARSE_INVERTED_INDEX",  # 稀疏向量专用倒排索引，适合 BM25 相似度计算
        metric_type="BM25",  # 指定为 BM25 相似度计算相关性分数
    )

    # ====================================================
    # 第四步：创建Collection
    # ====================================================
    client.create_collection(
        collection_name=COLLECTION_NAME,  # 指定Collection名称
        schema=schema,  # 指定Collection的Schema
        index_params=index_params,  # 指定Collection的索引参数
    )


def get_sample_documents() -> List[dict]:
    """准备测试数据，并生成对应的稠密向量"""
    documents = [
        {"text": "AB-120 是降噪耳机型号。", "source": "产品型号.md"},
        {"text": "降噪耳机退货后，退款通常三个工作日内到账。", "source": "退款规则.md"},
        {"text": "订单发货后可以在订单详情查看物流轨迹。", "source": "物流规则.md"},
    ]

    # 批量生成稠密向量（仅对正文进行 Embedding）
    dense_vectors = embedding_model.embed_documents([document["text"] for document in documents])

    # 组装插入 Milvus 的数据
    insert_data = []

    # 让每篇文档与自己的向量一一对应
    for document, dense_vector in zip(documents, dense_vectors):
        # 把当前实体加入写入列表
        insert_data.append({
            "text": document["text"],  # 写入原始正文
            "source": document["source"],  # 写入文档来源
            "dense_vector": dense_vector,  # 写入真实稠密向量
        })

    return insert_data


def search_hybrid(client: MilvusClient, question: str) -> List[List[dict]]:
    """执行混合搜索（向量检索 + BM25 检索）"""

    # 将用户查询转为稠密向量（注意：查询使用 embed_query，文档入库使用 embed_documents）
    query_dense_vector = embedding_model.embed_query(question)

    # ---------------- 构造两路检索请求 ----------------
    # 1. 稠密向量语义检索
    dense_request = AnnSearchRequest(
        data=[query_dense_vector],  # 传入用户问题的稠密向量
        anns_field="dense_vector",  # 指定稠密向量字段
        param={
            "metric_type": "COSINE",  # 使用余弦相似度比较语义
            "param": {}
        },
        limit=10  # 向量检索最多返回10个候选
    )

    # 2. BM25 关键词检索（直接传原始文本，Milvus 内部自动分词并转换）
    sparse_request = AnnSearchRequest(
        data=[question],  # 直接传入原始查询文本
        anns_field="sparse_vector",
        param={
            "metric_type": "BM25",  # 指定为 BM25 相似度计算相关性分数
            "param": {}
        },
        limit=10  # BM25检索最多返回10个候选
    )

    # ---------------- 执行混合检索 ----------------
    # 同时执行两路检索并融合结果
    return client.hybrid_search(
        collection_name=COLLECTION_NAME,  # 指定Collection名称
        reqs=[sparse_request, dense_request],  # 第一个是稀疏（BM25），第二个是稠密（向量）

        # --- 方案 A：使用加权融合 (WeightedRanker) ---
        # 如果倾向某一个检索结果，则使用加权Ranker融合，倾向哪个检索结果则写哪个权重大
        # 这里的 0.9 对应 sparse_request (BM25)，0.1 对应 dense_request (向量)
        ranker=WeightedRanker(
            0.9,  # BM25检索
            0.1,  # 向量检索
        ),

        # --- 方案 B：使用 RRF 倒数排名融合 (RRFRanker，推荐使用，无需调参) ---
        # ranker=RRFRanker(
        #     k=60  # RRF 平滑常数，通常设为 60，用于平滑排名差距，默认值为 60
        # ),
        limit=5,  # 最终返回5个候选
        output_fields=["text", "source"],  # 只返回文本和来源字段
    )


if __name__ == '__main__':
    # 连接 Milvus
    client = MilvusClient(url=MILVUS_URI, db_name=MILVUS_DB_NAME)

    # 1. 建表
    create_collection(client)

    # 2. 插入数据
    client.insert(
        collection_name=COLLECTION_NAME,
        data=get_sample_documents(),
    )

    # 3. 刷盘（确保数据可见）并将 Collection 加载到内存（搜索前必须执行）
    client.flush(collection_name=COLLECTION_NAME)
    client.load_collection(collection_name=COLLECTION_NAME)

    # 4. 执行搜索
    results = search_hybrid(client, "AB-120 退货后多久到账？")

    # 5. 格式化打印结果
    print("\n=== 混合检索结果 ===")
    if results and len(results) > 0:
        for hits in results:
            for hit in hits:
                print(f"[{hit['entity']['source']}] {hit['entity']['text']} (得分: {hit['distance']:.4f})")
