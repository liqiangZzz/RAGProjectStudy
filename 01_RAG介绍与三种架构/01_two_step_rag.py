from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from env_utils import DASHSCOPE_API_KEY
from init_llm import deepseek_llm_flash

# 1. 初始化嵌入模型
embedding_model = DashScopeEmbeddings(
    model="text-embedding-v1",
    dashscope_api_key=DASHSCOPE_API_KEY,
)

# 2. 准备知识内容
documents = [
    Document(
        page_content="退货规则：收货后七日内可以申请无理由退货，商品、包装和配件需要保持完整。",
        metadata={"source": "售后规则.md"},
    ),
    Document(
        page_content="退款规则：仓库验收通过后发起原路退款，通常三个工作日内到账。",
        metadata={"source": "退款规则.md"},
    ),
    Document(
        page_content="物流规则：订单发货后可以在订单详情查看物流轨迹。",
        metadata={"source": "物流规则.md"},
    ),
]

# 3.切分chunk
splitter = RecursiveCharacterTextSplitter(
    chunk_size=80,
    chunk_overlap=10,
    separators=["。", "；", "、"],
)

chunks = splitter.split_documents(documents)

print("chunks:", chunks)

# 4. 将 chunk 转换成 vector
vector_store = InMemoryVectorStore(embedding_model)
vector_store.add_documents(chunks)

# 5. 搜索 vector
retriever = vector_store.as_retriever(search_kwargs={"k": 1})
retriever_document = retriever.invoke("退货多久到账?")
print("retriever_document:", retriever_document)

content = "\n\n".join([document.page_content for document in retriever_document])


messages =[
    SystemMessage(content="你是一个专业的售后规则助手，根据我给你的资料回答用户问题。"),
    HumanMessage(content=f"参考资料：{content},用户问题：退货多久到账?")
   ]


# 6. 调用 LLM,模型回复
response = deepseek_llm_flash.invoke(content)
print("response:", response)
print(response.content)