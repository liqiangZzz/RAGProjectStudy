from typing import List

from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_milvus import Milvus
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pymilvus import MilvusClient, DataType

from env_utils import DASHSCOPE_API_KEY
from init_llm import deepseek_llm_flash

# ========== 全局配置 ==========
MILVUS_URI = "http://127.0.0.1:19530"
MILVUS_DB = "default"
COLLECTION_NAME = "rag_milvus_crud3"
DATA_FILE = "data/售后规则.txt"

# ========== 初始化全局 Embedding 模型 ==========
embedding_model = DashScopeEmbeddings(
    model="text-embedding-v1",
    dashscope_api_key=DASHSCOPE_API_KEY,
)


def load_and_split_documents(file_path: str) -> List[Document]:
    """第一步 & 第二步：加载文档并切分"""
    print("\n" + "=" * 25 + " 1. 加载与切分文档 " + "=" * 25)

    loader = TextLoader(file_path, encoding="utf-8")
    documents = loader.load()

    # 给原始文档注入业务元数据，所有切分出的 chunk 都会继承
    documents[0].metadata.update(
        {"product": "降噪耳机", "section": "退货与退款", "version": "2026-09"}
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=40,
        chunk_overlap=8,
        separators=["\n\n", "\n", "。", "；", "，", ""],
        keep_separator="end"
    )

    chunks = splitter.split_documents(documents)
    print(f"✅ 文档切分完成，共切分为 {len(chunks)} 个片段。")

    for i, chunk in enumerate(chunks, 1):
        print(f"  [片段 {i}]: {chunk.page_content}")

    return chunks


def build_milvus_collection(chunks: List[Document]) -> None:
    """第三步：原生建表与数据插入 (性能最好)"""
    print("\n" + "=" * 25 + " 2. 建表与数据插入 " + "=" * 25)

    client = MilvusClient(uri=MILVUS_URI, db_name=MILVUS_DB)

    try:
        # 1. 清理旧集合
        if client.has_collection(COLLECTION_NAME):
            client.drop_collection(COLLECTION_NAME)
            print(f"⚠️ 已删除旧的集合：{COLLECTION_NAME}")

        # 2. 获取向量维度
        dimension = len(embedding_model.embed_query("测试文本"))

        # 3. 定义 Schema
        schema = client.create_schema(auto_id=True, enable_dynamic_field=False)
        schema.add_field("id", DataType.INT64, is_primary=True)
        schema.add_field("page_content", DataType.VARCHAR, max_length=1000)
        schema.add_field("product", DataType.VARCHAR, max_length=1000)
        schema.add_field("section", DataType.VARCHAR, max_length=1000)
        schema.add_field("version", DataType.VARCHAR, max_length=1000)
        schema.add_field("vector", DataType.FLOAT_VECTOR, dim=dimension)

        # 4. 构建索引
        index_params = client.prepare_index_params()
        index_params.add_index(
            field_name="vector",
            index_type="AUTOINDEX",
            metric_type="COSINE",
        )

        # 5. 创建集合
        client.create_collection(
            collection_name=COLLECTION_NAME,
            schema=schema,
            index_params=index_params
        )
        print(f"✅ 建表完成：{COLLECTION_NAME} | 向量维度：{dimension}")

        # 6. 组装并插入数据
        data = [
            {
                "page_content": chunk.page_content,
                "product": chunk.metadata["product"],
                "section": chunk.metadata["section"],
                "version": chunk.metadata["version"],
                # 注意：这里使用 embed_documents 替代循环单条 embed_query，效率更高
                "vector": vector
            }
            for chunk, vector in zip(
                chunks,
                embedding_model.embed_documents([c.page_content for c in chunks])
            )
        ]

        client.insert(collection_name=COLLECTION_NAME, data=data)
        client.flush(collection_name=COLLECTION_NAME)
        client.load_collection(collection_name=COLLECTION_NAME)
        print(f"✅ 数据插入并加载完成，共插入 {len(data)} 条记录。")

    finally:
        # 确保释放原生 Client 连接
        client.close()


def search_documents(question: str, score_threshold: float = 0.3) -> None:
    """第四步：使用 LangChain 包装器进行检索"""
    print("\n" + "=" * 25 + " 3. 执行检索 " + "=" * 25)

    vector_store = Milvus(
        embedding_function=embedding_model,
        collection_name=COLLECTION_NAME,
        connection_args={"uri": MILVUS_URI, "db_name": MILVUS_DB},
        auto_id=True,
        primary_field="id",
        text_field="page_content",
        vector_field="vector",
        enable_dynamic_field=False,
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

    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": 3,
            "score_threshold": score_threshold,  # 建议使用 0.3 左右，0.7 对于此类短文本相似度往往过高
        },
    )

    docs = retriever.invoke(question)

    print(f"\n>>> 查询内容: '{question}' | 相似度阈值: >= {score_threshold}")
    print(f">>> 检索结果 (共 {len(docs)} 条):")

    if not docs:
        print("   ⚠️ 未找到符合条件的文档。阈值可能太高，建议适当调低。")
    else:
        for i, doc in enumerate(docs, 1):
            print(f"  [{i}] 内容: {doc.page_content}")
            print(f"      元数据: {doc.metadata}")

    all_info = []
    for document in docs:
        all_info.append(f"检索相关内容:{document.page_content},检索来源：{document.metadata.get("source")}")

    content = "\n\n".join(all_info)

    messages = []
    messages.append(SystemMessage(content=f"根据检索到的内容回答用户问题,检索到的内容如下：{content},回答结果中给我加上引用来源"))
    messages.append(HumanMessage(content=f"用户问题：{question}"))

    result = deepseek_llm_flash.invoke(messages)
    print("result:",result.content)

if __name__ == '__main__':
    # 1. 准备数据
    split_chunks = load_and_split_documents(DATA_FILE)

    # 2. 建库插入
    build_milvus_collection(split_chunks)

    # 3. 执行检索 (注意：阈值越大检索的文档内容越相关)
    search_documents(question="退货后退款多久到账？", score_threshold=0.9)