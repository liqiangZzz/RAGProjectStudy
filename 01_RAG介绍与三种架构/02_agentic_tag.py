from langchain.agents import create_agent
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import create_retriever_tool
from langchain_core.vectorstores import InMemoryVectorStore

from env_utils import DASHSCOPE_API_KEY
from init_llm import deepseek_llm_flash

# 1. 初始化嵌入模型
embedding_model = DashScopeEmbeddings(
    model="text-embedding-v1",
    dashscope_api_key=DASHSCOPE_API_KEY,
)

# 2.准备知识库
vector_store = InMemoryVectorStore(embedding_model)
vector_store.add_documents([
    Document(
        page_content="退货规则：收货后七日内可以申请无理由退货，商品、包装和配件需要保持完整。",
        metadata={"source": "售后规则.md"},
    ),
    Document(
        page_content="退款规则：仓库验收通过后发起原路退款，通常三个工作日内到账。",
        metadata={"source": "退款规则.md"},
    ),
])

# 3. 将 retriever 包装成 agent 可以调用的工具
retriever_tool = create_retriever_tool(
    retriever=vector_store.as_retriever(search_kwargs={"k": 1}),
    name="retriever",
    description="检索退款和退货相关内容，需要调用该工具",
    document_prompt=PromptTemplate.from_template("{page_content}\n来源：{source}"),
)

# 创建 agent
agent = create_agent(
    model=deepseek_llm_flash,
    tools=[retriever_tool],
)


result = agent.invoke({"messages":[{"role":"user","content":"退货有哪些要求？"}]})
print("result:", result)
print(result["messages"][-1].content)



