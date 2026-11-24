# ==============================================================================
# 固定检索节点的自纠错 RAG
# ==============================================================================
#  工作流定义
#START -> rewrite_node -> retrieve_node -> grade_node
#                                           ├── 相关 -> generate_node -> END
#                                           ├── 不相关(attempts<1) -> increment_attempts -> rewrite_node
#                                           └── 超次 -> fallback_node -> END

from typing import TypedDict

from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.documents import Document
from langchain_milvus import Milvus
from langgraph.constants import END, START
from langgraph.graph import StateGraph

from env_utils import DASHSCOPE_API_KEY
from init_llm import deepseek_llm_flash

# ==============================================================================
# 1. 向量数据库（Milvus）与 Embedding 模型配置
# ==============================================================================
MILVUS_URI = "http://localhost:19530"
MILVUS_DB_NAME = "default"
COLLECTION_NAME = "rag_course_self_correcting"

# 初始化阿里百炼（DashScope）向量嵌入模型
embedding_model = DashScopeEmbeddings(
    model="text-embedding-v1",
    dashscope_api_key=DASHSCOPE_API_KEY,
)

# 准备示例售后知识库文档（每条包含正文和来源 metadata）
documents = [
    Document(page_content="AB-120 耳机退货规则：收货后七日内可以申请无理由退货，需要保持商品和包装完整。",
             metadata={"source": "耳机退货.md"}),
    Document(page_content="AB-120 耳机退款规则：仓库验收退回商品后发起原路退款，通常三个工作日内到账。",
             metadata={"source": "耳机退款.md"}),
    Document(page_content="AB-120 蓝牙连接说明：长按功能键三秒进入配对模式。",
             metadata={"source": "耳机连接.md"}),
]

# 将文档向量化并写入 Milvus 向量库
vector_store = Milvus.from_documents(
    documents=documents,
    embedding=embedding_model,
    collection_name=COLLECTION_NAME,  # Milvus 集合名称
    connection_args={"uri": MILVUS_URI, "db_name": MILVUS_DB_NAME},  # 连接参数
    drop_old=True,  # 每次运行重建集合，避免旧数据干扰
    auto_id=True,  # 自动生成主键 ID
    index_params={"index_type": "AUTOINDEX", "metric_type": "COSINE", "params": {}},  # 自动索引，余弦相似度
    search_params={"metric_type": "COSINE", "params": {}},  # 搜索参数，余弦相似度
)

# 构建检索器：使用 MMR（最大边际相关性）算法，兼顾相关性与结果多样性
retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 2,  # 最终返回最相关且不重复的 2 个文档
        "fetch_k": 5,  # 先从向量库召回 5 个候选文档，再从中筛选出 2 个
    }
)


# ==============================================================================
# 2. 定义图状态（State）
# ==============================================================================
# total=False 表示初始化 State 时无需传入所有字段，各节点按需读取和更新即可
class RAGState(TypedDict, total=False):
    question: str  # 用户原始问题
    retrieve_query: str  # 经过改写后用于检索的查询语句
    documents: list[Document]  # 检索到的文档列表
    relevant: bool  # 检索文档与问题是否相关
    attempts: int  # 已重试（重新改写检索）的次数
    answer: str  # 最终生成的回答
    trace: list[str]  # 调试用的执行轨迹列表


# ==============================================================================
# 3. 定义图的各个节点（Node）与条件路由函数（Router）
# ==============================================================================
def rewrite_node(state: RAGState) -> dict:
    """节点 1：问题改写，将口语化的用户问题转化为更适合知识库检索的查询"""

    response = deepseek_llm_flash.invoke(
        f"把下面的售后问题改写成适合检索耳机知识相关的一行查询，只输出查询，原始问题：{state['question']}，如果原始问题不是售后问题，直接输出原始问题"
    )
    return {
        "retrieve_query": response.content,
        "trace": [*state.get("trace", []), f"rewrite_node：改写问题为{response.content}"]
    }


def retrieve_node(state: RAGState) -> dict:
    """节点 2：知识库检索，根据改写后的查询从 Milvus 中召回相关文档"""

    documents = retriever.invoke(state["retrieve_query"])

    return {
        "documents": documents,
        "trace": [*state.get("trace", []), f"retrieve_node：检索到{len(documents)}条文档"]
    }


def grade_node(state: RAGState) -> dict:
    """节点 3：文档相关性评估，判断检索到的内容是否足以回答当前问题"""

    document_content = "\n\n".join(
        f"内容：{doc.page_content},来源：{doc.metadata['source']}" for doc in state["documents"])
    response = deepseek_llm_flash.invoke(
        f"请判断下面的文档是否与用户问题相关，只输出YES或NO，不输出其他内容。用户问题：{state['retrieve_query']}\n检索资料内容：\n{document_content}"
    )
    # 解析模型输出，判断是否以 YES 开头
    relevant = response.content.strip().upper().startswith("YES")

    return {
        "relevant": relevant,
        "trace": [*state.get("trace", []), f"grade_node：判断结果为{relevant}"]
    }


def router_after_grade(state: RAGState) -> str:
    """路由函数：根据相关性评估结果及重试次数，决定工作流下一步分支"""

    # 1. 文档相关 -> 直接进入回答生成节点
    if state["relevant"]:
        return "generate_node"

    # 2. 文档不相关，且重试次数未超过上限（最多重试 1 次）-> 增加计数并回到改写检索流程
    if state.get("attempts", 0) < 1:
        return "increment_attempts"

    # 3. 超过重试次数仍不相关 -> 走兜底节点
    return "fallback_node"


def generate_node(state: RAGState) -> dict:
    """节点 4：答案生成，结合检索到的参考文档生成最终回复"""

    context = "\n\n".join(
        [f"内容：{document.page_content},来源：{document.metadata['source']}" for document in state["documents"]])
    response = deepseek_llm_flash.invoke(
        f"请下面的文档生成一个符合要求的答案，用户问题：{state['retrieve_query']}\n检索资料内容：\n{context}"
    )
    return {
        "answer": response.content,
        "trace": [*state.get("trace", []), f"generate_node：生成答案为{response.content}"]
    }


def increment_attempts(state: RAGState) -> dict:
    """节点 5（计数节点）：在触发重检索前，将重试次数 attempts 加 1"""

    attempts = state.get("attempts", 0) + 1

    return {
        "attempts": attempts,
        "trace": [*state.get("trace", []), f"increase_attempts：增加尝试次数为{attempts}"]
    }


def fallback_node(state: RAGState) -> dict:
    """节点 6（兜底节点）：多次尝试后依然无法检索到相关资料时，给出无法回答的提示"""
    return {
        "answer": "无法回答用户问题",
        "trace": [*state.get("trace", []), f"fallback_node：无法回答用户问题"]
    }


# ==============================================================================
# 4. 构建并编译 LangGraph 工作流
# ==============================================================================
builder = StateGraph(RAGState)

# 注册所有节点
builder.add_node("rewrite_node", rewrite_node)
builder.add_node("retrieve_node", retrieve_node)
builder.add_node("grade_node", grade_node)
builder.add_node("generate_node", generate_node)
builder.add_node("increment_attempts", increment_attempts)
builder.add_node("fallback_node", fallback_node)

# 连接主流程边：开始 -> 改写 -> 检索 -> 评估
builder.add_edge(START, "rewrite_node")
builder.add_edge("rewrite_node", "retrieve_node")
builder.add_edge("retrieve_node", "grade_node")

# 连接条件分支边：评估后根据路由结果分流
builder.add_conditional_edges(
    "grade_node",
    router_after_grade,
    {
        "generate_node": "generate_node",
        "increment_attempts": "increment_attempts",
        "fallback_node": "fallback_node"
    }
)
# 重试循环边：计数 +1 后重新回到改写节点
builder.add_edge("increment_attempts", "rewrite_node")
# 两条结束路径
builder.add_edge("generate_node", END)
builder.add_edge("fallback_node", END)

# 编译生成可执行图
graph = builder.compile()

# ==============================================================================
# 5. 导出流程图并运行测试
# ==============================================================================
# 将编译后的图结构保存为 Mermaid PNG 图片，便于可视化查看循环与分支结构
png_data = graph.get_graph().draw_mermaid_png()
with open("./images/01_fixed_retrieval_self_correcting_rag.png", "wb") as f:
    f.write(png_data)
print("图片已保存到 ./images/01_fixed_retrieval_self_correcting_rag.png")

# 测试用例 1：知识库内存在的售后问题（正常走 generate_node）
result = graph.invoke({"question": "耳机坏了，怎么退货？钱什么时候到账", "attempts": 0, "trace": []})
# 测试用例 2：与知识库无关的问题（触发重试后走 fallback_node）
# result = graph.invoke({"question": "北京天气如何？", "attempts": 0, "trace": []})

# 打印节点执行轨迹与最终回答
for item in result["trace"]:
    print(item)

print("\n\n\n最终回答：", result["answer"])
