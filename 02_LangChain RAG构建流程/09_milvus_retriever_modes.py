from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.documents import Document
from langchain_milvus import Milvus
from pymilvus import MilvusClient, DataType

from env_utils import DASHSCOPE_API_KEY

MILVUS_URI = "http://127.0.0.1:19530"
MILVUS_DB = "default"
COLLECTION_NAME = "rag_milvus_crud2"


def print_step(title: str):
    print("\n" + "=" * 25 + f" {title} " + "=" * 25)


def print_docs(title: str, docs: list):
    """格式化打印检索结果"""
    print(f"\n>>> {title} (共检索到 {len(docs)} 条)")
    if not docs:
        print("   未找到符合条件的结果")
        return
    for i, doc in enumerate(docs, 1):
        print(f"  [{i}] 内容: {doc.page_content}")
        print(f"      元数据: {doc.metadata}")


def main():
    # ========== 1. 初始化客户端与模型 ==========
    client = MilvusClient(uri=MILVUS_URI, db_name=MILVUS_DB)
    embedding_model = DashScopeEmbeddings(
        model="text-embedding-v1",
        dashscope_api_key=DASHSCOPE_API_KEY,
    )

    # 清理旧表
    if client.has_collection(COLLECTION_NAME):
        client.drop_collection(COLLECTION_NAME)
        print(f"已删除旧集合：{COLLECTION_NAME}")

    # ====================================================
    # 2. 原生建表与插入数据 (推荐，性能最好)
    # ====================================================
    print_step("1. 建表与数据插入")
    dimension = len(embedding_model.embed_query("测试文本"))

    # 注意：auto_id=True 时，插入数据不要包含 id 字段
    schema = client.create_schema(auto_id=True, enable_dynamic_field=False)
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field("page_content", DataType.VARCHAR, max_length=1000)
    schema.add_field("source", DataType.VARCHAR, max_length=1000)
    schema.add_field("product", DataType.VARCHAR, max_length=1000)
    schema.add_field("vector", DataType.FLOAT_VECTOR, dim=dimension)  # 向量字段

    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="vector",
        index_type="AUTOINDEX",
        metric_type="COSINE", # 在向量检索时使用的相似度度量方式为“余弦相似度”
    )

    client.create_collection(
        collection_name=COLLECTION_NAME,
        schema=schema,
        index_params=index_params
    )
    print("建表完成：", COLLECTION_NAME, "| 向量维度：", dimension)

    # 准备数据
    documents = [
        Document(page_content="耳机退货需要保持包装完整。", metadata={"source": "耳机退货.md", "product": "耳机"}),
        Document(page_content="耳机退款通常三个工作日内到账。", metadata={"source": "耳机退款.md", "product": "耳机"}),
        Document(page_content="手表激活后不支持无理由退货。", metadata={"source": "手表退货.md", "product": "手表"}),
    ]

    vectors = embedding_model.embed_documents([doc.page_content for doc in documents])

    data = [
        {
            "page_content": doc.page_content,
            "source": doc.metadata["source"],
            "product": doc.metadata["product"],
            "vector": vector,
        } for doc, vector in zip(documents, vectors)
    ]

    client.insert(collection_name=COLLECTION_NAME, data=data)
    client.flush(COLLECTION_NAME)
    client.load_collection(COLLECTION_NAME)
    print(f"成功插入 {len(data)} 条数据并加载至内存。")

    # ====================================================
    # 3. 挂载 LangChain 包装器
    # ====================================================
    # 将原生的 Client 包装成 LangChain 方便使用的 VectorStore
    vector_store = Milvus(
        embedding_function=embedding_model,  # 用于将查询文本转换为向量
        collection_name=COLLECTION_NAME,
        connection_args={"uri": MILVUS_URI, "db_name": MILVUS_DB},
        auto_id=True,   # 主键自动生成
        primary_field="id", # 主键字段名
        text_field="page_content",   # 文本字段名
        vector_field="vector",# 向量字段名
        enable_dynamic_field=False,  # 关闭动态字段

        # 【索引参数说明】
        # 告诉 LangChain 该集合建表时使用的是什么索引（AUTOINDEX）和度量方式（COSINE）。
        # 核心作用：当你在上层使用 similarity_score_threshold（分数阈值过滤）时，
        # LangChain 会根据这里的信息，把底层查询出的“距离”正确换算成“相关性分数（Score）”。
        index_params={"index_type": "AUTOINDEX", "metric_type": "COSINE"},

        # 【检索参数说明】
        # 告诉底层的 pymilvus 在执行 search 检索时，使用什么度量方式。
        # 注意：必须与上面的 index_params 以及数据库实际的索引保持一致，否则会报错或导致检索结果错乱。
        search_params={"metric_type": "COSINE"},
    )

    # ====================================================
    # 4. 三种检索模式测试
    # ====================================================

    # 模式 1：基础相似度 (Top-1)
    print_step("模式 1：Similarity (K=1),")
    similarity_retriever = vector_store.as_retriever(search_kwargs={"k": 1})
    print_docs("基础相似度", similarity_retriever.invoke("耳机退款多久到账?"))

    # 模式 2：带分数阈值的相似度 (过滤掉分数太低的)
    print_step("模式 2：Similarity Score Threshold (K=2, 阈值=0.7)")
    threshold_retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": 5,
            "score_threshold": 0.7,  # 因为用的是 COSINE，0.7 算是一个较高的要求
        },
    )
    print_docs("带分数阈值过滤", threshold_retriever.invoke("退款多久到账？"))

    # 模式 3：MMR (最大边际相关性)
    print_step("模式 3：MMR (兼顾相关性与多样性)")
    # MMR 检索模式：过滤条件使用 Milvus 的 expr 语法
    mmr_retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 2,  # 最终返回 2 条
            "fetch_k": 3,  # 从 Milvus 拉取 3 条候选
            # 注意：底层 Milvus 引擎只认原生布尔表达式（如 expr 字符串），
            # 无法识别 LangChain 通用的 filter 字典，因此这里必须用 Milvus 专属语法。
            "expr": "product == '手表'",
        }
    )
    print_docs("MMR 检索 (指定 product='手表')", mmr_retriever.invoke("退货规则"))


if __name__ == '__main__':
    main()