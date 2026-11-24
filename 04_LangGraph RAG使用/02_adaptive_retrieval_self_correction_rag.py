# ==============================================================================
# 自适应检索的自纠错 RAG
# ==============================================================================
# 工作流定义
# START -> call_model
#           ├── 有 tool_calls -> tools (ToolNode) -> grade_documents
#           │                                         ├── 相关 -> answer -> END
#           │                                         ├── 不相关(attempts<1) -> rewrite_and_retrieve -> grade_documents
#           │                                         └── 超次 -> fallback -> END
#           └── 无 tool_calls -> END（不检索，直接回答）

from typing import TypedDict, Annotated

from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.documents import Document
from langchain_core.messages import AnyMessage, SystemMessage, AIMessage, ToolMessage, HumanMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import create_retriever_tool
from langchain_milvus import Milvus
from langgraph.constants import START, END
from langgraph.graph import add_messages, StateGraph
from langgraph.prebuilt import ToolNode

from env_utils import DASHSCOPE_API_KEY
from init_llm import deepseek_llm_flash

# ==============================================================================
# 1. 向量数据库（Milvus）与 Embedding 模型配置
# ==============================================================================
MILVUS_URI = "http://localhost:19530"
MILVUS_DB_NAME = "default"
COLLECTION_NAME = "rag_course_ch4_self_correcting"

# 初始化阿里百炼（DashScope）向量嵌入模型
embedding_model = DashScopeEmbeddings(
    model="text-embedding-v1",
    dashscope_api_key=DASHSCOPE_API_KEY,
)

# 将示例售后知识库文档向量化并写入 Milvus
vector_store = Milvus.from_documents(
    [
        Document("AB-120 耳机退货规则：收货后七日内可以申请无理由退货，需要保持商品和包装完整。",
                 metadata={"source": "耳机退货.md"}),
        Document("AB-120 耳机退款规则：仓库验收退回商品后发起原路退款，通常三个工作日内到账。",
                 metadata={"source": "耳机退款.md"}),
        Document("AB-120 蓝牙连接说明：长按功能键三秒进入配对模式。", metadata={"source": "耳机连接.md"}),
    ],
    embedding_model,
    collection_name=COLLECTION_NAME,  # Milvus 集合名称
    connection_args={"uri": MILVUS_URI, "db_name": MILVUS_DB_NAME},  # 连接参数
    drop_old=True,  # 每次运行是否删除旧集合
    auto_id=True,  # 是否自动分配 ID
    index_params={"index_type": "AUTOINDEX", "metric_type": "COSINE", "params": {}},  # 索引参数
    search_params={"metric_type": "COSINE", "params": {}},  # 搜索参数
)

# ==============================================================================
# 2. 创建检索工具并绑定到 LLM
# ==============================================================================
# 将向量库的 retriever 封装为 LangChain 工具，供模型自主调用
retrieve_tool = create_retriever_tool(
    retriever=vector_store.as_retriever(search_kwargs={"k": 2}),
    name="retrieve_tool",
    description="检索 AB-120 耳机的退货、退款和连接说明。无需改写任何参数，直接调用就可以",
    document_prompt=PromptTemplate.from_template("{page_content}\n来源：{source}"),
)

# 将工具绑定到模型（注意 bind_tools 需要传入列表）
model_with_tools = deepseek_llm_flash.bind_tools([retrieve_tool])


# ==============================================================================
# 3. 定义图状态（State）
# ==============================================================================
class State(TypedDict, total=False):
    # add_messages 是 LangGraph 内置的 reducer，会将新消息追加到历史消息列表中，而不是直接覆盖
    messages: Annotated[list[AnyMessage], add_messages]  # 对话及工具调用消息历史
    question: str  # 用户原始问题
    relevant: bool  # 检索结果是否与问题相关
    attempts: int  # 已尝试“问题改写并重新检索”的次数
    trace: list[str]  # 记录各节点执行轨迹，方便调试观察流程


# ==============================================================================
# 4. 定义图的各个节点（Node）与条件路由函数（Router）
# ==============================================================================
def call_model(state: State) -> State:
    """节点 1：调用绑定了检索工具的模型，决定是直接回复还是发起工具检索"""
    prompt = f"你是 AB-120 耳机售后助手。涉及退货、退款、到账、连接问题时必须调用检索工具。用户原始问题：{state['question']}"
    response = model_with_tools.invoke([SystemMessage(content=prompt), *state["messages"]])
    return {
        "messages": [response],
        "trace": [*state.get("trace", []), f"call_model(tool_calls={len(response.tool_calls or [])})"]
    }


def router_after_model(state: State) -> str:
    """路由 1：根据模型输出中是否包含 tool_calls，决定走工具节点还是直接结束"""
    last = state["messages"][-1]

    # 如果模型发起了工具调用，路由到 "tools" 节点执行检索
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tools"
    # 否则说明模型已直接回答完毕，结束流程
    return "end"


# 判断检索资料是否相关
def grade_documents(state: State) -> dict:
    """节点 2：评估检索回来的资料是否足以回答用户的原始问题（Self-RAG / CRAG 核心步骤）"""

    # 从消息列表中取出最近一条 ToolMessage 的检索内容
    tool_messages = [message for message in state["messages"] if isinstance(message, ToolMessage)]
    tool_content = tool_messages[-1].content if tool_messages else "没有检索结果。"
    response = deepseek_llm_flash.invoke(
        "判断检索资料能否回答用户问题，只输出 YES 或 NO。\n"
        f"用户问题：{state['question']}\n检索资料：\n{tool_content}"
    )
    result = response.content.strip().upper()
    relevant = result.startswith("YES")
    return {
        "relevant": relevant,
        "trace": [*state.get("trace", []), f"grade_documents({result})"]
    }


def router_after_grade(state: State) -> str:
    """路由 2：根据相关性评估结果决定下一步走向"""

    # 资料相关 -> 进入生成回答节点
    if state["relevant"]:
        return "answer"

    # 资料不相关，但重试次数未达上限（最多改写重试 1 次） -> 进入问题改写与重检索节点
    if state.get("attempts", 0) < 1:
        return "rewrite_and_retrieve"

    # 超过重试次数仍未找到有效资料 -> 进入兜底节点
    return "fallback"


def answer(state: State) -> dict:
    """节点 3：结合检索到的资料生成最终回答"""

    # 从消息列表中取出最近一条 ToolMessage 的检索内容
    tool_messages = [message for message in state["messages"] if isinstance(message, ToolMessage)]
    tool_content = tool_messages[-1].content if tool_messages else "没有检索结果。"
    response = deepseek_llm_flash.invoke(
        "你是售后助手。只根据资料回答，保留时间限定，回答末尾列出来源。\n"
        f"用户问题：{state['question']}\n资料：\n{tool_content}"
    )
    return {
        "messages": [response],
        "trace": [*state.get("trace", []), f"answer({response.content})"]
    }


def rewrite_and_retrieve(state: State) -> dict:
    """节点 4（纠错循环）：当初次检索效果不佳时，改写用户查询并重新检索向量库"""

    response = deepseek_llm_flash.invoke(
        "把下面的售后问题改写成更适合检索 AB-120 耳机知识库的一行查询，只输出查询。\n"
        f"原始问题：{state['question']}"
    )
    query = response.content.strip().replace("\n", " ")

    # 使用改写后的查询直接检索 Milvus
    documents = vector_store.similarity_search(query, k=2)
    tool_content = "\n".join(f"{doc.page_content}\n来源：{doc.metadata['source']}" for doc in documents)
    return {
        # 将新检索到的内容封装为 ToolMessage 追加到消息中，供下一轮 grade_documents 评估
        "messages": [ToolMessage(content=tool_content, tool_call_id="rewrite-retrieval")],
        "attempts": state.get("attempts", 0) + 1,  # 尝试次数 +1
        "trace": [*state.get("trace", []), f"rewrite_and_retrieve({query})"]
    }


def fallback(state: State) -> dict:
    """节点 5（兜底）：检索改写后依然无法回答时，提示转人工客服"""
    return {
        "messages": [AIMessage(content="资料不足，请转人工客服。")],
        "trace": [*state.get("trace", []), "fallback"]
    }


# ==============================================================================
# 5. 构建并编译 LangGraph 工作流
# ==============================================================================
builder = StateGraph(State)

# 注册所有节点
builder.add_node("call_model", call_model)
builder.add_node("tools", ToolNode([retrieve_tool]))
builder.add_node("grade_documents", grade_documents)
builder.add_node("rewrite_and_retrieve", rewrite_and_retrieve)
builder.add_node("answer", answer)
builder.add_node("fallback", fallback)

# 定义节点间的边与条件边
builder.add_edge(START, "call_model")
builder.add_conditional_edges(
    "call_model",
    router_after_model,
    {"tools": "tools", "end": END}
)
builder.add_edge("tools", "grade_documents")
# 连接条件分支边：评估后根据路由结果分流
builder.add_conditional_edges(
    "grade_documents",
    router_after_grade,
    {
        "answer": "answer",
        "rewrite_and_retrieve": "rewrite_and_retrieve",
        "fallback": "fallback"
    }
)
builder.add_edge("rewrite_and_retrieve", "grade_documents")  # 重检索后再次回到相关性评估
builder.add_edge("answer", END)
builder.add_edge("fallback", END)

# 编译生成可执行的图对象
graph = builder.compile()

# ==============================================================================
# 6. 导出流程图并运行测试
# ==============================================================================
# 将工作流结构导出为 Mermaid PNG 图片，方便直观查看状态流转
png_data = graph.get_graph().draw_mermaid_png()
with open("./images/02_adaptive_retrieval_self_correction_rag.png", "wb") as f:
    f.write(png_data)
print("图片已保存到 ./images/02_adaptive_retrieval_self_correction_rag.png")

# 发起一次测试提问
question = "耳机坏了想退，钱什么时候回来？"
result = graph.invoke(
    {
        "messages": [HumanMessage(content=question)],
        "question": question,
        "attempts": 0,
        "trace": []
    }
)

print("用户问题：", question)
print("执行轨迹：")
for item in result["trace"]:
    print("-", item)
print("\n最终回答：", result["messages"][-1].content)
