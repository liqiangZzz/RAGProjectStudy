# ============================================================
# 使用 Milvus 作为向量库的 create_agent RAG Agent
# 知识库持久化到 Milvus，模型通过 Retriever Tool 自主决定是否检索。
# ============================================================

from langchain.agents import create_agent
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import create_retriever_tool
from langchain_milvus import Milvus

from env_utils import DASHSCOPE_API_KEY
from init_llm import deepseek_llm_flash

# ============================================================
# 第一步：连接 Milvus 并使用真实 Embedding 建立售后知识库
# ============================================================
MILVUS_URI = "http://localhost:19530"
MILVUS_DB_NAME = "default"
COLLECTION_NAME = "rag_course_create_agent_milvus"


def build_vector_store() -> Milvus:
    embedding_model = DashScopeEmbeddings(
        model="text-embedding-v1",
        dashscope_api_key=DASHSCOPE_API_KEY,
    )

    documents = [
        Document(
            page_content="退款规则：仓库验收后发起原路退款，通常三个工作日内到账。",
            metadata={"source": "退款规则.md"},
        ),
        Document(
            page_content="物流规则：订单发货后可以在订单详情查看物流轨迹。",
            metadata={"source": "物流规则.md"},
        ),
    ]

    return Milvus.from_documents(
        documents=documents,  # 待写入的 Document 列表
        embedding=embedding_model,  # 用于将文档转换为向量的模型
        collection_name=COLLECTION_NAME,  # 指定要挂载的集合名
        connection_args={"uri": MILVUS_URI, "db_name": MILVUS_DB_NAME},  # 连接 Milvus 的参数
        drop_old=True,  # 每次运行都从干净集合开始
        auto_id=True,  # 主键由 Milvus 自动生成
        index_params={
            "index_type": "AUTOINDEX",
            "metric_type": "COSINE",
            "params": {}
        },  # 索引配置：AUTOINDEX + COSINE
        search_params={
            "metric_type": "COSINE",
            "params": {},
        }  # 检索参数为 COSINE
    )


# ============================================================
# 第二步：把 Milvus Retriever 包装成 Agent 工具
# ============================================================
vector_store = build_vector_store()

document_prompt = PromptTemplate.from_template("{page_content}\n来源：{source}")

retriever_tool = create_retriever_tool(
    retriever=vector_store.as_retriever(search_kwargs={"k": 1}),
    name="retrieve_after_sale_rules",
    description="检索退款到账和物流规则。遇到退款、到账或物流问题必须调用。",
    document_prompt=document_prompt,
)

if __name__ == '__main__':
    # ========================================================
    # 第三步：使用 create_agent 创建 Agent
    # ========================================================
    agent = create_agent(
        model=deepseek_llm_flash,
        tools=[retriever_tool],
        system_prompt=(
            "你是电商售后 Agent。涉及退款、到账或物流问题时必须先调用检索工具。"
            "只能根据工具结果回答，回答末尾列出来源。"
        )
    )

    # ========================================================
    # 第四步：调用 Agent 并观察消息循环
    # ========================================================
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "退款多久到账?"}]}
    )
    for msg in result["messages"]:
        msg.pretty_print()

    print("*" * 100)
    print("\n最终回答：", result["messages"][-1].content)
