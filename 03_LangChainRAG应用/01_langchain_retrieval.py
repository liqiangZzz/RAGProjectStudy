# 查询改写、过滤、MMR检索和引用回答
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from env_utils import DASHSCOPE_API_KEY
from init_llm import deepseek_llm_flash

embedding_model = DashScopeEmbeddings(
    model="text-embedding-v1",
    dashscope_api_key=DASHSCOPE_API_KEY,
)

if __name__ == '__main__':
    # ========================================================
    # 第一步：准备多产品知识库并切分
    # ========================================================
    documents = [
        Document(
            page_content="降噪耳机退货：收货后七日内可以申请无理由退货，包装和配件需要保持完整。",
            metadata={"source": "耳机售后.md", "product": "降噪耳机"},
        ),
        Document(
            page_content="降噪耳机退款：仓库验收后发起原路退款，通常三个工作日内到账。",
            metadata={"source": "耳机退款.md", "product": "降噪耳机"},
        ),
        Document(
            page_content="智能手表规则：手表激活后不支持无理由退货。",
            metadata={"source": "手表售后.md", "product": "智能手表"},
        ),

    ]

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=15,
        separators=["\n\n", "\n", "。", "，", "！", "？", ""]
    )

    chunks = text_splitter.split_documents(documents)
    print(f"✅ 文档切分完成，共切分 {len(chunks)} 个 chunk。")

    # 存入向量库
    vector_store = InMemoryVectorStore(embedding_model)
    vector_store.add_documents(chunks)

    # ========================================================
    # 第二步：使用LLM模型完成改写查询
    # ========================================================

    question = "这款降噪耳机退掉后，钱回到银行卡需要多久？"

    rewrite_prompt = f"""
      你是一个专业的售后知识库检索专家。请将用户的原始问题进行改写，以便更好地检索相关资料。
      要求：
      1. 提取核心产品实体（如：降噪耳机）。
      2. 保留用户的核心意图（如：退款到账时间）。
      3. 生成 3 条不同表达方式的自然语言短句，不要只堆砌关键词。
      4. 每条查询占一行，不要包含序号。

      原始问题：{question}
      """

    rewrite_response = deepseek_llm_flash.invoke(rewrite_prompt)

    rewritten_query = rewrite_response.content.strip().replace("\n", " ")
    print(f"改写后的查询：{rewritten_query}")
    print("----------------" * 5)

    # ========================================================
    # 第三步：使用千问Embedding、MMR和产品过滤检索
    # ========================================================
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 2,
            "fetch_k": 3,
            "filter": lambda document: document.metadata.get("product") == "降噪耳机",
        },
    )

    retrieved_documents = retriever.invoke(rewritten_query)

    # 组织召回内容
    content = "\n".join(
        f"来源：{doc.metadata['source']}\n内容：{doc.page_content}"
        for doc in retrieved_documents
    )

    # ========================================================
    # 第四步：直接构造消息并调用LLM
    # ========================================================

    messages = [
        SystemMessage(content="你是一个专业的售后规则助手，根据我给你的资料回答用户问题。"),
        HumanMessage(content=f"参考资料：{content},用户问题：{question}"),
    ]
    response = deepseek_llm_flash.invoke(messages)
    answer = response.content

    print("原始问题:", question)
    print("改写查询:", rewritten_query)
    print("召回来源：", [document.metadata["source"] for document in retrieved_documents])
    print("\n最终回答：", answer)
