from typing import List

from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore

from env_utils import DASHSCOPE_API_KEY

# ========== 1. 初始化嵌入模型 ==========
embedding_model = DashScopeEmbeddings(
    model="text-embedding-v1",
    dashscope_api_key=DASHSCOPE_API_KEY,
)
def print_docs(title: str, docs: List[Document]) -> None:
    """辅助函数：格式化打印检索结果，方便直观对比差异"""
    print(f"\n{'='*15} {title} {'='*15}")
    if not docs:
        print("未检索到结果")
        return
    for i, doc in enumerate(docs, 1):
        print(f"[{i}] 内容: {doc.page_content}")
        print(f"    元数据: {doc.metadata}")


if __name__ == "__main__":
    # ========================================================
    # 第一步：创建千问语义向量库
    # ========================================================
    documents = [
        Document("耳机退货需要保持包装完整。", metadata={"source": "耳机退货.md", "product": "耳机"}),
        Document("耳机退款通常三个工作日内到账。", metadata={"source": "耳机退款.md", "product": "耳机"}),
        Document("手表激活后不支持无理由退货。", metadata={"source": "手表退货.md", "product": "手表"}),
    ]

    vector_store = InMemoryVectorStore(embedding_model)
    # 写入文档到向量库
    vector_store.add_documents(documents)

    # ========================================================
    # 第二步：基础检索：相似度检索 (Similarity Search)
    # ========================================================
    query = "耳机退货需要保持包装完整？"

    # 1. 使用 as_retriever 包装接口
    base_retriever = vector_store.as_retriever(search_kwargs={"k": 1})
    retriever_results = base_retriever.invoke(query)
    print_docs("基础 Retriever 检索 (Top-1)", retriever_results)

    # 2. 直接调用底层方法（ 和as_retriever检索结果一致）
    similarity_results = vector_store.similarity_search(query, k=1)
    print_docs("直接 similarity_search 检索 (Top-1)", similarity_results)


    # ========================================================
    # 第三步：MMR 检索：兼顾相关性与多样性
    # 【关键优化】过滤条件改为标准字典格式，这样迁移到 Milvus 等数据库时绝对不会报错
    # ========================================================
    mmr_retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 2,  # 返回 2 个结果
            "fetch_k": 3,  # 先取出 3 个最相似的作为候选池（必须 >= k）
            # "lambda_mult": 0.5, # 可选：控制多样性，1 为纯相似度，0 为纯多样性，默认 0.5
            "filter": lambda doc: doc.metadata.get("product") == "耳机"
        },
    )
    mmr_results = mmr_retriever.invoke("耳机退货和退款规则")
    print_docs("MMR 检索 (Top-2，限定 product='耳机')", mmr_results)
