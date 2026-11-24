import operator
import os
from typing import List, TypedDict, Annotated

import dashscope
from langchain_community.document_compressors import DashScopeRerank
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.documents import Document
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from langgraph.constants import START, END
from langgraph.graph import add_messages, StateGraph
from pydantic import BaseModel, Field
from pymilvus import MilvusClient, DataType, Function, FunctionType, AnnSearchRequest, RRFRanker

from env_utils import DASHSCOPE_API_KEY
from init_llm import deepseek_llm_flash

MILVUS_URI = "http://localhost:19530"
MILVUS_DB_NAME = "default"
COLLECTION_NAME = "rag_course_native"

# ==============================================================================
# 1. 初始化 Embedding 与 Rerank 模型
# ==============================================================================
# 注意：数据入库（建库）与用户提问检索（查询）必须使用同一个 Embedding 模型，否则向量空间不一致
embedding_model = DashScopeEmbeddings(
    model="text-embedding-v1",
    dashscope_api_key=DASHSCOPE_API_KEY,
)

# 精排模型（Cross-Encoder）：对粗排召回的多篇候选文档重新打分，只保留最相关的 top_n 篇
reranker = DashScopeRerank(
    client=dashscope.TextReRank,  # 显式传入 DashScope 客户端，避免部分版本报错
    model="gte-rerank-v2",
    top_n=3,  # 返回前3个文档
    api_key=DASHSCOPE_API_KEY,
)

# ==============================================================================
# 2. 原生 Milvus 建表、建索引与数据写入
# ==============================================================================
def build_collection_natively(client: MilvusClient, dimension: int) -> None:
    """原生创建支持稠密向量 + BM25 稀疏向量的混合检索 Collection。"""

    if client.has_collection(COLLECTION_NAME):
        client.drop_collection(COLLECTION_NAME)

    # 开启动态字段，source 可作为动态字段写入并自动回到 Document.metadata
    schema = client.create_schema(auto_id=True, enable_dynamic_field=True)
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True, auto_id=True)

    # 文本字段：开启分词器，供 BM25 算法计算词频/逆文档频率
    schema.add_field(
        field_name="text",
        datatype=DataType.VARCHAR,
        max_length=1000,
        enable_analyzer=True,
        analyzer_params={"tokenizer": "jieba"}
    )

    # 元数据字段：用于展示或后续的 expr 过滤
    schema.add_field("source", DataType.VARCHAR, max_length=100)
    schema.add_field("product", DataType.VARCHAR, max_length=100)

    # 稠密向量：由 Embedding 模型生成
    schema.add_field("dense_vector", DataType.FLOAT_VECTOR, dim=dimension)
    # 稀疏向量：由 Milvus 内部 BM25 函数自动生成
    schema.add_field("sparse_vector", DataType.SPARSE_FLOAT_VECTOR)

    # 注册 Milvus 原生 BM25 转换函数：写入 text 时自动生成对应的 sparse_vector
    schema.add_function(
        Function(
            name="bm25_function",
            function_type=FunctionType.BM25,
            input_field_names=["text"],  # 输入文本
            output_field_names=["sparse_vector"],  # 自动输出稀疏向量
        )
    )

    # 定义索引
    index_params = client.prepare_index_params()

    # 稠密向量索引：使用余弦相似度（COSINE）衡量语义相近程度
    index_params.add_index(
        field_name="dense_vector",
        index_type="AUTOINDEX",
        metric_type="COSINE",
    )

    # 稀疏向量索引：倒排索引 + BM25 度量，擅长精准匹配专有名词/型号/数字（如 "AB-120"）
    index_params.add_index(
        field_name="sparse_vector",
        index_type="SPARSE_INVERTED_INDEX",
        metric_type="BM25",
    )

    client.create_collection(
        collection_name=COLLECTION_NAME,
        schema=schema,
        index_params=index_params,
    )


def insert_documents(client: MilvusClient) -> None:
    """批量生成稠密向量，并将售后规则文档写入 Milvus。"""

    rows = [
        {
            "text": "退款规则：仓库验收后发起原路退款，通常三个工作日内到账。",
            "source": "退款规则.md",
            "product": "通用",
        },
        {
            "text": "物流规则：订单发货后可以在订单详情查看物流轨迹。",
            "source": "物流规则.md",
            "product": "通用",
        },
        {
            "text": (
                "AB-120耳机售后规则：因耳机属于贴身电子产品，一经拆封，若无质量问题，"
                "不支持七天无理由退换货。若存在音质故障等质量问题，支持15天内免费换新。"
            ),
            "source": "AB120专属规则.md",
            "product": "AB-120",
        },
        {
            "text": (
                "AB-120耳机保修规则：自签收之日起，提供为期一年的免费保修服务。"
                "非人为损坏（如进水、摔坏、私自拆机）不在免费保修范围内。"
            ),
            "source": "AB120专属规则.md",
            "product": "AB-120",
        },
        {
            "text": (
                "发票规则：本店支持开具电子普通发票和增值税专用发票。订单确认收货后，"
                "可在订单详情页申请开具，电子发票通常会在24小时内发送至您的预留邮箱。"
            ),
            "source": "发票规则.md",
            "product": "通用",
        },
        {
            "text": (
                "退换货运费规则：因商品质量问题产生的退换货，由本店承担来回运费；"
                "因买家个人原因（如不喜欢、拍错）产生的退货，运费由买家自行承担。"
                "若订单含运费险，可理赔部分金额。"
            ),
            "source": "运费规则.md",
            "product": "通用",
        },
        {
            "text": (
                "退货包装要求：退回的商品请务必保持原包装完整（包含耳机本体、充电线、"
                "说明书等所有配件），如有赠品需一并退回，否则可能会被仓库拒收，影响退款进度。"
            ),
            "source": "退货规则.md",
            "product": "通用",
        },
        {
            "text": (
                "客服介入规则：若您与商家在售后处理上产生争议（如退款金额未达成一致、"
                "商家拒绝退换），您可以点击页面上的“申请平台客服介入”，平台将在48小时内介入处理。"
            ),
            "source": "客服规则.md",
            "product": "通用",
        },
    ]

    # 批量将文本转为稠密向量；sparse_vector 无需传入，由 Milvus 服务端自动生成
    vectors = embedding_model.embed_documents([row["text"] for row in rows])

    client.insert(
        collection_name=COLLECTION_NAME,
        data=[{**row, "dense_vector": vector} for row, vector in zip(rows, vectors)]
    )

    # 刷新 Collection 状态，确保数据写入完成
    client.flush(collection_name=COLLECTION_NAME)

    # 加载 Collection，确保索引生效
    client.load_collection(collection_name=COLLECTION_NAME)


milvus_client = MilvusClient(uri=MILVUS_URI, db_name=MILVUS_DB_NAME)


# ==============================================================================
# 3. 定义图状态（State）与结构化输出模型
# ==============================================================================
class State(TypedDict, total=False):
    """LangGraph 全局状态定义（未在此声明的字段会被图自动丢弃）。"""

    # add_messages 是 LangGraph 内置的 reducer，会将新消息追加到历史消息列表中，而不是直接覆盖
    messages: Annotated[list[AnyMessage], add_messages]

    question: str                   # 用户原始提问
    sub_queries: list[str]          # 改写分解后的多个子查询
    retrieved_docs: list[Document]  # 混合检索（粗排）召回的候选文档
    reranked_docs: list[Document]   # Reranker（精排）筛选后的核心文档
    relevant: bool                  # 相关性评估结果（True=足够回答，False=需重写或兜底）
    attempts: int                   # 检索重写已尝试次数（防止死循环）
    fallback_message: str           # 兜底提示文本
    generation: str                 # 最终生成的回答文本

    # operator.add：自动将节点返回的列表追加到已有 trace 中（节点只需返回新增日志列表即可）
    trace: Annotated[list[str], operator.add]


class SubQueriesOutput(BaseModel):
    """查询改写与分解的结构化输出"""
    sub_queries: List[str] = Field(
        description="将用户的复杂问题分解为多个独立、适合用于检索的简短子查询"
    )


class GradeOutput(BaseModel):
    """文档相关性评估的结构化输出"""

    relevant: bool = Field(description="检索到的参考文档是否包含回答用户问题所需的相关信息")


def format_docs(docs: list[Document]) -> str:
    """将 Document 列表格式化为清晰的上下文文本。"""
    return "\n\n".join(
        f"[{i + 1}] 来源：{doc.metadata.get('source', '未知')} "
        f"(商品：{doc.metadata.get('product', '通用')})\n"
        f"内容：{doc.page_content}"
        for i, doc in enumerate(docs)
    )


# ==============================================================================
# 4. 定义 LangGraph 各工作流节点
# ==============================================================================
def rewrite_and_decompose(state: State) -> dict:
    """节点 1 - 改写与分解：根据用户问题生成多个独立检索子查询。"""

    sub_queries_model = deepseek_llm_flash.with_structured_output(SubQueriesOutput)
    user_question = state.get("question", "").strip()

    prompt = (
        "请将以下用户的复杂问题拆解为 2-3 个适合在电商售后知识库中检索的独立短查询。\n"
        f"用户问题：{user_question}"
    )

    multi_query_result = sub_queries_model.invoke(prompt)
    queries = [q.strip() for q in multi_query_result.sub_queries if q.strip()] or [user_question]
    return {
        "sub_queries": queries,
        "trace": [f"rewrite_and_decompose: 拆解为 {len(queries)} 个子查询 -> {queries}"],
    }


def hybrid_retrieve(state: State) -> dict:
    """节点 2 - 混合检索（粗排召回）：商品+通用过滤 -> 稠密向量/BM25 混合检索 -> RRF 融合。"""

    sub_queries = [q.strip() for q in state.get("sub_queries", []) if q.strip()]

    if not sub_queries:
        sub_queries = [state.get("question", "").strip()]

    all_documents = []
    seen_texts = set()  # 用于去重
    filter_expr = 'product in ["AB-120", "通用"]'

    # 2. 遍历每个子查询进行混合检索
    for query in sub_queries:
        # 生成当前子查询的稠密向量
        query_vec = embedding_model.embed_query(query)

        # 定义密集向量检索请求
        dense_search = AnnSearchRequest(
            data=[query_vec],  # 注意：data 这里需要是列表形式
            anns_field="dense_vector",
            param={"metric_type": "COSINE", "params": {}},
            limit=10,  # 召回 10 条
            expr=filter_expr
        )

        # 定义 BM25 稀疏检索请求
        bm25_search = AnnSearchRequest(
            data=[query],  # data 是纯文本
            anns_field="sparse_vector",
            param={"metric_type": "BM25", "params": {}},
            limit=10,  # 召回 10 条
            expr=filter_expr
        )

        # 执行混合检索 【RRF（倒数排序融合）：综合两路排名，兼顾语义召回与字面精准匹配】
        hybrid_results = milvus_client.hybrid_search(
            collection_name=COLLECTION_NAME,
            reqs=[dense_search, bm25_search],
            ranker=RRFRanker(k=60),
            output_fields=["text", "source", "product"],
            limit=5,
        )

        # 3. 解析结果并去重
        for hits in hybrid_results:  # hybrid_results 是一个 list，里面包着 hit 列表
            for hit in hits:
                text = hit["entity"]["text"]
                if text not in seen_texts:
                    seen_texts.add(text)
                    # 转换为 LangChain 的 Document 对象
                    all_documents.append(
                        Document(
                            page_content=text,
                            metadata={
                                "source": hit["entity"].get("source", "未知"),
                                "product": hit["entity"].get("product", "未知"),
                                "score": hit["distance"]  # RRF 融合后的得分
                            }
                        )
                    )

    return {
        "retrieved_docs": all_documents,
        "trace": [f"hybrid_retrieve: 检索了 {len(sub_queries)} 个子查询，去重后召回 {len(all_documents)} 篇文档"]
    }


def rerank(state: State) -> dict:
    """节点 3 - 重排（精排）：利用 DashScope Reranker 筛选最相关的 top_n 文档。"""
    documents = state.get("retrieved_docs", [])  # 从状态中获取检索到的文档
    if not documents:
        return {
            "reranked_docs": [],
            "trace": ["rerank: 没有文档可重排"],
        }

    question = state.get("question", "")

    # 利用 DashScope Reranker 筛选最相关的 top_n 文档
    reranked_documents = list(
        reranker.compress_documents(
            documents=documents,
            query=question,
        )
    )
    return {
        "reranked_docs": reranked_documents,
        "trace": [f"rerank: 从 {len(documents)} 篇文档中精排出 {len(reranked_documents)} 篇"],
    }


def grade(state: State) -> dict:
    """节点 4 - 检索评估：判断重排后的文档是否足以回答用户问题。"""

    reranked_documents = state.get("reranked_docs", [])
    if not reranked_documents:
        return {
            "relevant": False,
            "trace": ["grade: 无重排文档可评估，判定为不相关"],
        }

    question = state.get("question", "")
    grader_model = deepseek_llm_flash.with_structured_output(GradeOutput)
    docs_text = format_docs(reranked_documents)

    result = grader_model.invoke(
        "请评估以下参考资料是否包含回答用户问题所需的有效信息。\n"
        f"用户问题：{question}\n\n"
        f"参考资料：\n{docs_text}"
    )

    return {
        "relevant": result.relevant,
        "trace": [f"grade: 评估了 {len(reranked_documents)} 篇文档，相关性={result.relevant}"],
    }


def router_after_node(state: State) -> str:
    """路由决策：相关则生成，不相关且未超重试次数则重写，否则兜底。"""

    if state.get("relevant", False):
        return "generate"

    if state.get("attempts", 0) < 1:
        return "retry_rewrite"

    # 兜底：如果以上条件都不满足，返回 fallback 路由
    return "fallback"


def generate(state: State) -> dict:
    """节点 5A - 生成节点：根据用户问题，生成回答"""
    question = state.get("question", "")
    reranked_documents = state.get("reranked_docs", [])
    docs_text = format_docs(reranked_documents)

    response = deepseek_llm_flash.invoke(
        "你是电商售后助手。请严格只根据提供的参考资料回答用户问题，注意保留资料中的时间限定，"
        "并在回答末尾列出引用到的来源文件名。\n\n"
        f"用户问题：{question}\n\n"
        f"参考资料：\n{docs_text}"
    )
    return {
        "generation": [response.content],
        "messages": [response],
        "trace": [f"generate: 已生成最终回答（长度 {len(response.content)} 字）"],
    }


def retry_rewrite(state: State) -> dict:
    """节点 5B - 重试重写节点：结合已有结果换角度重新拆解查询词。"""

    question = state.get("question", "")
    reranked_documents = state.get("reranked_docs", [])
    docs_text = format_docs(reranked_documents) if reranked_documents else "暂无有效资料"

    sub_queries_model = deepseek_llm_flash.with_structured_output(SubQueriesOutput)
    result = sub_queries_model.invoke(
        "此前的检索未找到足够相关的售后规则。请换个角度或补充同义关键词，"
        "将用户问题重新拆解为 2-3 个适合在知识库中检索的独立短查询。\n"
        f"用户问题：{question}\n"
        f"上次检索到的参考文档：\n{docs_text}"
    )

    queries = [q.strip() for q in result.sub_queries if q.strip()] or [question]
    new_attempts = state.get("attempts", 0) + 1

    return {
        "sub_queries": queries,
        "attempts": new_attempts,
        "trace": [f"retry_rewrite: 第 {new_attempts} 次重试，重写子查询为 {queries}"],
    }


def fallback(state: State) -> dict:
    """节点 5C -兜底节点：多次检索仍无法满足时，返回人工客服引导。"""
    msg = "当前知识库资料不足以完整回答您的问题，请点击页面上的“申请平台客服介入”或联系人工客服处理。"
    return {
        "fallback_message": msg,
        "generation": msg,
        "messages": [AIMessage(content=msg)],
        "trace": [f"fallback: {msg}"],
    }


# ==============================================================================
# 5. 构建并编译 LangGraph 工作流
# ==============================================================================
builder = StateGraph(State)

# 注册所有节点
builder.add_node("rewrite_and_decompose", rewrite_and_decompose)
builder.add_node("hybrid_retrieve", hybrid_retrieve)
builder.add_node("rerank", rerank)
builder.add_node("grade", grade)
builder.add_node("generate", generate)
builder.add_node("retry_rewrite", retry_rewrite)
builder.add_node("fallback", fallback)

# 定义固定流转边
builder.add_edge(START, "rewrite_and_decompose")
builder.add_edge("rewrite_and_decompose", "hybrid_retrieve")
builder.add_edge("hybrid_retrieve", "rerank")
builder.add_edge("rerank", "grade")

# 定义条件分支边：评估 -> 生成 / 重试 / 兜底
builder.add_conditional_edges(
    "grade",
    router_after_node,
    {
        "generate": "generate",
        "retry_rewrite": "retry_rewrite",
        "fallback": "fallback"
    }
)
builder.add_edge("retry_rewrite", "hybrid_retrieve")  # 形成「重写 -> 重检 -> 重排 -> 重评」闭环
builder.add_edge("generate", END)
builder.add_edge("fallback", END)

# 编译生成可执行的图对象
graph = builder.compile()

# ==============================================================================
# 6. 主入口：建库、导出流程图并运行测试
# ==============================================================================
if __name__ == "__main__":

    # 1. 初始化 Milvus 集合并写入测试文档
    dimension = len(embedding_model.embed_query("测试文本"))
    build_collection_natively(milvus_client, dimension=dimension)
    insert_documents(milvus_client)

    # 2. 导出工作流可视化结构图
    os.makedirs("./images", exist_ok=True)
    png_data = graph.get_graph().draw_mermaid_png()
    with open("./images/03_langgraph_hybrid_rag.png", "wb") as f:
        f.write(png_data)
    print("图片已保存到 ./images/03_langgraph_hybrid_rag.png")

    # 3. 发起测试提问并打印执行轨迹与结果
    question = "耳机坏了想退，钱什么时候回来？"
    result = graph.invoke(
        {
            "messages": [HumanMessage(content=question)],
            "question": question,
            "attempts": 0,
            "trace": [],
        }
    )

    print("用户问题：", question)
    print("执行轨迹：")
    for item in result["trace"]:
        print("-", item)
    print("\n最终回答：", result["messages"][-1].content)
