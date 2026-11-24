# =============================================================================
# 使用DashScopeRerank对候选文档重新排序
# Reranker把查询和候选文档一起送入模型，输出新的相关性分数
# =============================================================================

import dashscope
from langchain_community.document_compressors import DashScopeRerank
from langchain_core.documents import Document

from env_utils import DASHSCOPE_API_KEY


def main() -> None:
    # ========================================================
    # 第一步：准备候选文档
    # ========================================================
    documents = [
        Document(
            page_content="仓库验收通过后，款项通常三个工作日退回原支付渠道。",
            metadata={"doc_id": "refund-semantic"},
        ),
        Document(
            page_content="收货后七日内可以申请无理由退货，需要保证商品和包装完整。",
            metadata={"doc_id": "return-rule"},
        ),
        Document(
            page_content="AB-120 是降噪耳机型号，适用于蓝牙连接。",
            metadata={"doc_id": "model-ab120"},
        ),
        Document(
            page_content="订单发货后可以查看物流轨迹，超过48小时未更新联系客服。",
            metadata={"doc_id": "logistics"},
        ),
        Document(
            page_content="退款将按照原支付渠道原路退回。",
            metadata={"doc_id": "refund-channel"},
        ),
    ]

    question = "钱什么时候能退回银行卡？"

    print("Reranker输入顺序：")
    for rank, document in enumerate(documents, start=1):
        print(f"{rank}. {document.metadata['doc_id']}，{document.page_content}")

    # ========================================================
    # 第二步：创建DashScope Reranker
    # ========================================================
    reranker = DashScopeRerank(
        client=dashscope.TextReRank,  # 显式传入 DashScope 客户端，避免错误
        model="gte-rerank-v2",  # 模型名称
        top_n=3,  # 返回前3个文档
        api_key=DASHSCOPE_API_KEY
    )

    # ========================================================
    # 第三步：对候选文档重新评分和排序
    # ========================================================
    reranked_documents = reranker.compress_documents(
        documents=documents,
        query=question,
    )

    # ========================================================
    # 第四步：输出精排结果
    # ========================================================
    print("\nReranker精排结果：")
    for rank, document in enumerate(reranked_documents, start=1):
        print(
            f"{rank}. {document.metadata['doc_id']} "
            f"relevance={document.metadata['relevance_score']:.6f}"
        )
        print("   内容：", document.page_content)


if __name__ == '__main__':
    main()
