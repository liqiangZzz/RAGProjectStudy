from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore

from env_utils import DASHSCOPE_API_KEY

# 1. 初始化嵌入模型
embedding_model = DashScopeEmbeddings(
    model="text-embedding-v1",
    dashscope_api_key=DASHSCOPE_API_KEY,
)

# 2. 加载文档
documents = [
    Document(
        page_content="退款规则：仓库验收通过后发起原路退款，通常三个工作日内到账。",
        metadata={"source": "退款规则.md"},
    ),
    Document(
        page_content="物流规则：订单发货后可以在订单详情查看物流轨迹。",
        metadata={"source": "物流规则.md"},
    ),
]

# 3. 基于内存的向量数据库
vector_store = InMemoryVectorStore(embedding_model)
vector_store.add_documents(documents)

# 将文本转换为向量表示
# results = embedding_model.embed_query("退款什么时候到账？")
# print("results:", results)


retriever = vector_store.as_retriever(k=1)
results = retriever.invoke("退款什么时候到账？")
print("as_retriever:", results)


results = vector_store.similarity_search("退款什么时候到账？")
print("similarity_search:", results)


results = vector_store.similarity_search_with_score("退款什么时候到账？")
print("similarity_search_with_score:", results)
