# 演示使用Multi-Query、查询分解、HyDE、路由、重排和上下文压缩结果
from typing import Literal

from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from pydantic import BaseModel, Field

from env_utils import DASHSCOPE_API_KEY
from init_llm import deepseek_llm_flash

# ============================================================
# 第一步：构建Embedding模型
# ============================================================
embedding_model = DashScopeEmbeddings(
    model="text-embedding-v1",
    dashscope_api_key=DASHSCOPE_API_KEY,
)


# ============================================================
# 第二步：定义结构化输出格式
# ============================================================
# 用于 Multi-Query
class QueryList(BaseModel):
    queries: list[str] = Field(description="用于检索的查询列表")


#  定义新的结构化输出：允许大模型返回单个路由标签
class RouteResult(BaseModel):
    route: Literal["product", "after_sale", "logistics"] = Field(
        description="按相关性从高到低排列的路由标签，仅返回一个")


# 定义新的结构化输出：允许大模型返回多个路由标签
class MultiRouteResult(BaseModel):
    # 与单路由的区别：把单个字符串换成了列表，允许命中多个领域
    routes: list[Literal["product", "after_sale", "logistics"]] = Field(
        description="按相关性从高到低排列的路由标签列表，允许存在多个"
    )


# 用于 Reranker
class RerankResultWithIrrelevant(BaseModel):
    relevant_sources: list[str] = Field(
        description="按相关性从高到低排列的相关来源名称"
    )
    irrelevant_sources: list[str] = Field(
        description="与用户问题完全无关，无法回答该问题的来源名称"
    )


# ============================================================
# 第三步：创建真实语义知识库
# ============================================================
documents = [
    Document(
        "耳机退货：收货后七日内可以申请退货，包装和配件需要完整。",
        metadata={"source": "耳机售后.md", "route": "after_sale"},
    ),
    Document(
        "耳机退款：仓库验收后发起退款，通常三个工作日内到账。",
        metadata={"source": "耳机退款.md", "route": "after_sale"},
    ),
    Document(
        "产品型号：AB-120 是降噪耳机，AB-121 是运动耳机。",
        metadata={"source": "产品型号.md", "route": "product"},
    ),
    Document(
        "物流查询：订单发货后可以在订单详情查看物流轨迹。",
        metadata={"source": "物流规则.md", "route": "logistics"},
    ),
]

vector_store = InMemoryVectorStore(embedding_model)
vector_store.add_documents(documents)

if __name__ == '__main__':
    # ========================================================
    # 第四步：Multi-Query——真实模型生成多个检索表达
    # 假设用户问题为“耳机怎么退”
    # ========================================================
    multi_query_model = deepseek_llm_flash.with_structured_output(QueryList)

    multi_query_result = multi_query_model.invoke(
        "针对问题‘耳机怎么退’生成 3 个不同表达的中文检索查询。"
    )

    multi_query_documents = []  # 存储所有查询的文档
    seen_sources = set()  # 存储已处理的来源，避免重复添加

    for query in multi_query_result.queries:
        for document in vector_store.similarity_search(query, k=1):
            source = document.metadata["source"]

            # 避免重复添加
            if source in seen_sources:
                continue

            seen_sources.add(source)
            multi_query_documents.append(document)
    print("=" * 100)
    print("Multi-Query：", multi_query_result.queries[:3])
    print("Multi-Query召回：",
          "\t".join(f"内容：{doc.page_content}来源：{doc.metadata['source']}" for doc in multi_query_documents))

    print("=" * 100)
    # ========================================================
    # 第五步：查询分解——真实模型拆成两个子问题
    # ========================================================
    decomposition_result = multi_query_model.invoke("把问题‘耳机能否退货，退款多久到账’拆成2个独立中文子问题。")

    decomposition_documents = []
    seen_sources = set()

    for query in decomposition_result.queries:
        for document in vector_store.similarity_search(query, k=1):
            source = document.metadata["source"]

            # 避免重复添加
            if source in seen_sources:
                continue

            seen_sources.add(source)
            decomposition_documents.append(document)

    print("查询分解：", decomposition_result.queries[:2])
    print("查询分解召回：",
          "".join(f"\n\t内容：{doc.page_content}来源：{doc.metadata['source']}" for doc in decomposition_documents))

    print("=" * 100)

    # ========================================================
    # 第六步：HyDE——真实模型生成假设性答案并用它检索
    # ========================================================
    hyde_document = deepseek_llm_flash.invoke(
        "针对问题‘耳机怎么退’，生成一段不超过50字的假设性售后答案，用于向量检索。").content

    # 1. 直接用 similarity_search_with_score，拿到 (Document, score) 的元组列表
    docs_with_scores = vector_store.similarity_search_with_score(
        query=hyde_document,
        k=3,
    )
    # 2. 手动根据阈值过滤，InMemoryVectorStore 返回的是归一化后的余弦相似度
    hyde_documents = [
        doc for doc, score in docs_with_scores if score >= 0.5
    ]

    print("HyDE假设性答案：", hyde_document)
    print("HyDE假设性答案召回：",
          "".join(f"\n\t内容：{doc.page_content}来源：{doc.metadata['source']}" for doc in hyde_documents))

    print("=" * 100)

    # ========================================================
    # 第七步：Routing——真实模型选择知识库（单路由选择）
    # ========================================================

    routing_model = deepseek_llm_flash.with_structured_output(RouteResult)

    route_result = routing_model.invoke("问题：AB-120是什么型号？ 请判断该问题涉及了哪个领域，返回对应的路由标签。")

    # 1. 从 documents 中筛选出路由为 routing_result.route 的文档
    routed_documents = [
        document
        for document in documents
        if document.metadata["route"] == route_result.route
    ]

    routed_store = InMemoryVectorStore(embedding_model)
    routed_store.add_documents(routed_documents)

    route_documents = routed_store.similarity_search(
        "AB-120是什么型号？",
        k=1,
    )

    print("Single-Routing结果：", route_result.route)
    print("Single-Routing召回：",
          "".join(
              f"\n\t- [路由: {document.metadata['route']}] 来源：{document.metadata['source']} | 内容：{document.page_content}"
              for document in route_documents)
          )
    print("=" * 100)

    # ========================================================
    # 第八步：Multi-Routing (多路由) —— 跨领域知识库同时检索
    # ========================================================

    # 1. 定义新的结构化输出：允许大模型返回多个路由标签
    multi_routing_model = deepseek_llm_flash.with_structured_output(MultiRouteResult)

    # 2. 构造一个跨领域的问题（同时涉及：型号 + 退货 + 物流）
    multi_route_question = "我买的 AB-120 耳机如果申请退货，物流费谁来承担？"

    # 3. 大模型进行多路由判断
    multi_route_result = multi_routing_model.invoke(
        f"问题：{multi_route_question}。请判断该问题涉及了哪些领域，返回对应的路由标签。"
    )

    # 4. 根据多路由结果，收集所有相关的文档
    multi_routed_documents = [
        document
        for document in documents
        if document.metadata["route"] in multi_route_result.routes
    ]

    # 5. 创建临时向量库并检索
    multi_routed_store = InMemoryVectorStore(embedding_model)
    multi_routed_store.add_documents(multi_routed_documents)

    # 用原问题去这几个混合领域的知识库里查
    final_multi_route_docs = multi_routed_store.similarity_search(
        multi_route_question,
        k=5,
    )

    print("Multi-Routing 路由结果：", multi_route_result.routes)
    print("Multi-Routing 召回结果：",
          "".join(
              f"\n\t- [路由: {document.metadata['route']}] 来源：{document.metadata['source']} | 内容：{document.page_content}"
              for document in final_multi_route_docs)
          )

    print("=" * 100)
    # ========================================================
    # 第九步：Reranking——真实模型重新排列候选来源（带不相关剔除）
    # ========================================================

    # 使用新的结构化输出模型
    rank_model = deepseek_llm_flash.with_structured_output(RerankResultWithIrrelevant)

    candidates = vector_store.similarity_search("耳机退款多久到账？", k=5)

    candidate_text = "\n".join(
        f"来源={document.metadata['source']}，内容={document.page_content}"
        for document in candidates
    )
    # 明确要求区分相关与不相关
    rank_result = rank_model.invoke(
        "用户问题：耳机退款多久到账？\n"
        "请仔细阅读下面的候选资料：\n"
        "1. 将与退款到账时间相关的资料，按相关性从高到低放入 `relevant_sources`。\n"
        "2. 如果资料讲的是产品型号、物流发货，完全无法回答退款问题的，放入 `irrelevant_sources`。\n"
        f"候选资料：\n{candidate_text}"
    )

    source_map = {
        document.metadata["source"]: document
        for document in candidates
    }

    # 提取相关的文档（保持重排后的顺序）
    reranked_documents = [
        source_map[source]
        for source in rank_result.relevant_sources
        if source in source_map
    ]

    print("重排前：", [document.metadata["source"] for document in candidates])
    print("重排后（相关）：", [document.metadata["source"] for document in reranked_documents])
    print("被剔除（不相关）：", rank_result.irrelevant_sources)

    print("=" * 100)

    # ========================================================
    # 第十步：Contextual Compression——提取必要证据
    # ========================================================
    if not reranked_documents:
        print("⚠️ 警告：所有候选文档均被判为不相关，无需压缩。")
        compressed_context = "未找到相关信息。"
    else:
        content = "\n\n".join(
            f"【来源：{doc.metadata['source']}】\n内容：{doc.page_content}"
            for doc in reranked_documents
        )

        # 2. 强化 Prompt：明确要求保留来源，并告知大模型只针对提问提取
        compression_prompt = (
            "你是一个严谨的文档阅读理解助手。请从下面提供的资料中，提取出回答"
            "‘退款退货多久到账’以及‘产品来源’所需的原文信息。\n"
            "要求：\n"
            "1. 只提取资料中已有的内容，绝不编造或补充资料外内容。\n"
            "2. 提取时请务必保留对应的【来源】信息，以便后续溯源。\n"
            "3. 如果某篇资料完全没有提到相关内容，直接跳过。\n\n"
            f"资料：\n{content}"
        )

        compressed_context = deepseek_llm_flash.invoke(compression_prompt).content
    print("上下文压缩：", compressed_context)
