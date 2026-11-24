from typing import TypedDict

from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langgraph.constants import START, END
from langgraph.graph import StateGraph

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


class HybridState(TypedDict, total=False):
    original_question: str  # 用户原始问题
    retrieval_query: str  # 检索查询
    documents: list[Document]  # 检索到的知识库文档
    relevant: bool  # 是否相关
    answer: str  # 回答
    attempts: int  # 尝试次数
    grade_result: str  # 验证结果
    trace: list[str]  # 执行轨迹


# 检索节点
def retrieve(state: HybridState) -> HybridState:
    """检索节点"""
    documents = vector_store.as_retriever().invoke(state["retrieval_query"], k=2)

    return {"documents": documents, "trace": [*state.get("trace", []), "retrieve"]}


# 文档验证
def validate_documents(state: HybridState) -> HybridState:
    content = "\n\n".join(
        [document.page_content + "\n来源：" + document.metadata["source"] for document in state["documents"]])

    response = deepseek_llm_flash.invoke(
        f"请根据以上检索到的资料，判断是否能回答用户问题，只输出YES或NO。用户问题：{state['original_question']}，检索到的资料：{content}")
    grade_result = response.content.strip()
    relevant = grade_result.startswith("YES")

    return {
        "relevant": relevant,
        "grade_result": grade_result,
        "trace": [*state.get("trace", []), f"validate_documents({relevant})"]
    }


# 路由函数
def route_after_validation(state: HybridState):
    """路由函数"""
    if state["relevant"]:
        return "generate"

    if state.get("attempts", 0) <= 2:
        return "rewrite_question"

    return "fallback"


# generate 生成回答
def generate(state: HybridState) -> HybridState:
    """生成回答"""
    content = "\n\n".join(
        [document.page_content + "\n来源：" + document.metadata["source"] for document in state["documents"]])
    response = deepseek_llm_flash.invoke(f"只能根据给的资料回答问题。资料：{content}，用户问题：{state['retrieval_query']}")
    answer = response.content.strip()

    return {
        "answer": answer,
        "trace": [*state.get("trace", []), f"generate({answer})"]
    }


# rewrite_question 重写问题
def rewrite_question(state: HybridState) -> HybridState:

    """重写问题"""
    response = deepseek_llm_flash.invoke(
        f"请根据以上检索到的资料，回答用户问题。用户问题：{state['original_question']}，重写为一个更符合知识库的查询的问题，只输出重新写的问题，不要解释")
    rewritten = response.content.strip()

    return {
        "retrieval_query": rewritten,
        "attempts": state.get("attempts", 0) + 1,
        "trace": [*state.get("trace", []), f"rewrite_question({rewritten})"]
    }


# fallback 回退
def fallback(state: HybridState) -> HybridState:
    """回退"""

    return {
        "answer": "无法回答用户问题",
        "trace": [*state.get("trace", []), "fallback"]
    }


builder = StateGraph(HybridState)
builder.add_node("retrieve", retrieve)
builder.add_node("validate_documents", validate_documents)
builder.add_node("rewrite_question", rewrite_question)
builder.add_node("generate", generate)
builder.add_node("fallback", fallback)

builder.add_edge(START, "retrieve")
builder.add_edge("retrieve", "validate_documents")
builder.add_conditional_edges(
    "validate_documents",
    route_after_validation,
    {
        "generate": "generate",
        "rewrite_question": "rewrite_question",
        "fallback": "fallback"
    }
)
builder.add_edge("rewrite_question", "retrieve")
builder.add_edge("generate", END)
builder.add_edge("fallback", END)

graph = builder.compile()
# 保存 Mermaid 图为 PNG
png_data = graph.get_graph().draw_mermaid_png()

# wb表示二进制写入模式
with open("./images/03_hybrid_rag.png", "wb") as f:
    f.write(png_data)
print("图片已保存到 ./images/03_hybrid_rag.png")

result = graph.invoke({
    "original_question": "退货需要符合那些条件？",
    "retrieval_query": "退货需要符合那些条件？",
    "attempts": 0,
    "trace": []
})
print(result)
