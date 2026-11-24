# ==============================================================================
# RAG 进阶实战：混合检索 (BM25 + 向量) -> RRF融合 -> DashScope Rerank 精排
# ==============================================================================
from typing import List, Dict

import dashscope
from langchain_community.document_compressors import DashScopeRerank
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.documents import Document
from pymilvus import MilvusClient, DataType, FunctionType, Function, AnnSearchRequest, RRFRanker

from env_utils import DASHSCOPE_API_KEY

# 全局配置
MILVUS_URI = "http://localhost:19530"
MILVUS_DB_NAME = "default"
COLLECTION_NAME = "tmp_hybrid_search_rerank"


def init_collection(client: MilvusClient, dimension: int) -> None:
    """1. 建表：创建同时包含稠密向量、稀疏向量和文本字段的 Collection"""

    # 幂等设计：如果表已存在则删除，保证脚本可重复运行
    if client.has_collection(COLLECTION_NAME):
        client.drop_collection(COLLECTION_NAME)

    # 创建 Schema，禁止动态字段，主键自动生成
    schema = client.create_schema(auto_id=True, enable_dynamic_field=False)
    # 主键
    schema.add_field("id", DataType.INT64, is_primary=True)
    # 文本字段：开启分词器，供 BM25 算法计算词频/逆文档频率
    schema.add_field(
        field_name="text",
        datatype=DataType.VARCHAR,
        max_length=1000,
        enable_analyzer=True,
        analyzer_params={"tokenizer": "jieba"}
    )
    # 元数据字段：用于展示或后续的 expr 过滤
    schema.add_field("source", DataType.VARCHAR, max_length=100)
    # 稠密向量：由 Embedding 模型生成
    schema.add_field("dense_vector", DataType.FLOAT_VECTOR, dim=dimension)
    # 稀疏向量：由 Milvus 内部 BM25 函数自动生成
    schema.add_field("sparse_vector", DataType.SPARSE_FLOAT_VECTOR)

    # 注册 Milvus 原生 BM25 转换函数
    schema.add_function(
        Function(
            name="bm25_function",
            function_type=FunctionType.BM25,
            input_field_names=["text"],  # 输入文本
            output_field_names=["sparse_vector"], # 自动输出稀疏向量
        )
    )

    # 定义索引
    index_params = client.prepare_index_params()

    # 稠密向量：使用 AUTOINDEX 自动选优，COSINE 计算余弦相似度
    index_params.add_index(
        field_name="dense_vector",
        index_type="AUTOINDEX",
        metric_type="COSINE",
    )

    # 稀疏向量：使用倒排索引，指定使用 BM25 算法计算相关性
    index_params.add_index(
        field_name="sparse_vector",
        index_type="SPARSE_INVERTED_INDEX",
        metric_type="BM25",
    )

    # 创建 Collection
    client.create_collection(
        collection_name=COLLECTION_NAME,
        schema=schema,
        index_params=index_params,
    )


def insert_data(client: MilvusClient, embedding_model: DashScopeEmbeddings, documents: List[Dict]):
    """2. 数据入库：批量生成向量并插入 Milvus"""

    # 提取所有正文，批量调用 Embedding API（比单条调用快得多）
    texts = [doc["text"] for doc in documents]
    dense_vectors = embedding_model.embed_documents(texts)

    # 组装数据：将原始字段与计算好的向量合并
    insert_data = [
        {**doc, "dense_vector": vec}
        for doc, vec in zip(documents, dense_vectors)
    ]

    # 插入数据
    client.insert(COLLECTION_NAME, insert_data)

    # 刷盘并加载到内存（搜索前必须 load）
    client.flush(COLLECTION_NAME)
    client.load_collection(COLLECTION_NAME)


def search_hybrid(client: MilvusClient, embedding_model: DashScopeEmbeddings, query: str) -> List[dict]:
    """3. 混合检索（粗排召回）：执行向量+BM25检索，并用 RRF 融合"""

    # 将查询词转为稠密向量
    query_vec = embedding_model.embed_query(query)

    # 密集向量检索
    dense_results = AnnSearchRequest(
        data=[query_vec],
        anns_field="dense_vector",
        param={
            "metric_type": "COSINE",
            "params": {},
        },
        limit=10  # 召回 10 条
    )

    # BM25 检索
    sparse_results = AnnSearchRequest(
        data=[query], # 直接传入原始文本即可
        anns_field="sparse_vector",
        param={
            "metric_type": "BM25",
            "params": {},
        },
        limit=10 # 召回 10 条
    )

    # 执行混合检索
    # ranker=RRFRanker：使用倒数排名融合，规避了 COSINE 和 BM25 分数范围不一致的问题
    # limit=8：两路各召回 10 条，融合后只取最靠前的 8 条送入后续精排，节省资源
    hybrid_results = client.hybrid_search(
        collection_name=COLLECTION_NAME,
        reqs=[dense_results, sparse_results],
        ranker=RRFRanker(k=60),
        output_fields=["text", "source"],
        limit=8,
    )[0]
    return hybrid_results


def main():
    # 1. 初始化 Milvus 客户端 和 DashScope 模型
    client = MilvusClient(uri=MILVUS_URI, db_name=MILVUS_DB_NAME)
    embedding_model = DashScopeEmbeddings(model="text-embedding-v1", dashscope_api_key=DASHSCOPE_API_KEY)

    # 2. 准备数据：包含 2 条黄金文档，和 13 条强干扰项（关键词匹配但语义错误）
    mock_docs = [
        # 真正有用的黄金文档 (语义完全匹配)
        {"text": "退款通常会在审核通过后的三个工作日内原路退回至您的支付账户。", "source": "退款说明.md"},
        {"text": "退货签收并验收通过后，钱款将按原支付渠道退回，预计耗时 1-3 个工作日。", "source": "售后规则.md"},

        # 关键词高度匹配，但语义错误 (包含 AB-120 和 退款/钱，但讲的是别的)
        {"text": "AB-120 的内部批发价格属于商业机密，如需退款请联系大客户经理。", "source": "内部价格.md"},
        {"text": "AB-120 降噪耳机如果出现非人为损坏，支持免费维修，不涉及退款。", "source": "维修政策.md"},
        {"text": "AB-120 的定金在发货前可以全额退款。", "source": "预售规则.md"},
        {"text": "如果您购买的是 AB-120 配件，配件退款需单独联系客服处理。", "source": "配件说明.md"},

        # 其他干扰项 (包含退款、钱、时间等词，但场景不对)
        {"text": "信用卡退款的到账时间取决于发卡行，通常为 7-15 个工作日。", "source": "支付说明.md"},
        {"text": "订单发货后可以查看物流轨迹，快递超过 48 小时未更新请联系客服。", "source": "物流规则.md"},
        {"text": "本店支持七天无理由退货，但退货运费需要由买家自行承担。", "source": "退换规则.md"},
        {"text": "如果您使用优惠券下单，退款时优惠券将退回您的账户，不折现。", "source": "优惠券说明.md"},

        # 完全无关的噪音数据
        {"text": "AB-120 是降噪耳机型号，支持蓝牙 5.0 连接，续航 20 小时。", "source": "产品手册.md"},
        {"text": "请在干燥环境下存放耳机，避免进水导致内部短路。", "source": "产品保养.md"},
        {"text": "本店所有商品均提供正规发票，下单时请在备注中填写抬头。", "source": "发票说明.md"},
        {"text": "客服工作时间是周一至周五 9:00-18:00。", "source": "客服规范.md"},
        {"text": "AB-121 是运动耳机型号，具备 IPX5 级防水功能。", "source": "产品手册.md"},
    ]

    try:
        # 3. 建表并写入数据
        dimension = len(embedding_model.embed_query("测试"))
        init_collection(client, dimension)
        insert_data(client, embedding_model, mock_docs)
        print(f"✅ 成功写入 {len(mock_docs)} 条数据")

        # 4. 用户查询 (故意混合了"型号"和"退款"的语义)
        query = "AB-120 退货退款，钱什么时候能到账？"
        print(f"\n🔍 用户查询：{query}")

        # ========================================================
        # 阶段一：混合检索 (粗排召回)
        # ========================================================
        print("\n" + "=" * 60)
        print("🟡 阶段一：混合检索 (Hybrid Search) 召回 Top 8 候选")
        print("=" * 60)
        hybrid_results = search_hybrid(client, embedding_model, query)

        candidate_documents = []
        for i, hit in enumerate(hybrid_results, start=1):
            entity = hit["entity"]
            print(f"[Rank {i}] {entity['source']}")
            print(f"     {entity['text']}")

            # 将 Milvus 结果转为 LangChain Document
            candidate_documents.append(Document(
                page_content=entity["text"],
                metadata={"source": entity["source"], "origin_rank": i}
            ))

        # ========================================================
        # 阶段二：DashScope Rerank 精排
        # ========================================================
        print("\n" + "=" * 60)
        print("🟢 阶段二：Reranker 精排 (从 Top 8 中选出最相关的 Top 3)")
        print("=" * 60)

        reranker = DashScopeRerank(
            client=dashscope.TextReRank,
            model="gte-rerank-v2",
            top_n=3,
            api_key=DASHSCOPE_API_KEY
        )

        reranked_documents = reranker.compress_documents(
            documents=candidate_documents,
            query=query
        )

        for i, doc in enumerate(reranked_documents, start=1):
            score = doc.metadata.get('relevance_score', 0)
            print(f"[精排 Top {i}] 得分: {score:.4f} (原本在第 {doc.metadata['origin_rank']} 名)")
            print(f"     来源: {doc.metadata['source']}")
            print(f"     内容: {doc.page_content}")

    finally:
        pass
        # 5. 安全释放资源：无论代码是否报错，最后都会清理临时表
        # if client.has_collection(COLLECTION_NAME):
        #     client.drop_collection(COLLECTION_NAME)
        #     print("\n🧹 临时表已清理，环境安全")


if __name__ == "__main__":
    main()