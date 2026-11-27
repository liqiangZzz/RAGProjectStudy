"""NDCG@K（Normalized Discounted Cumulative Gain，归一化折损累计增益）评估示例。

指标含义：
    在「结果越靠前、权重越大（折损）」的前提下，衡量检索排序质量，
    并用「理想排序」做归一化，使取值范围落在 [0, 1]。

公式：
    DCG@K  = Σ (relevance_i / log2(i + 1))，i 从 1 开始（排名越靠后，分母越大，贡献被折损）
    IDCG@K = 理想排序下的 DCG@K（所有相关来源都排最前）
    NDCG@K = DCG@K / IDCG@K

参数分析：
    relevance_i 是排在第 i 位的那条结果的相关性打分，取值为 1.0 或 0.0。
    log2(i + 1) 是 NDCG 公式中的“位置折损因子”。它利用对数函数增长缓慢的特性，科学地惩罚那些被排在后面的正确文档，强制模型必须把最相关的资料排在最前面。
    log2(i + 1) 算的是「2 的多少次方等于 (i + 1)」（例如 i=1 时 2^1=2 得 1，i=3 时 2^2=4 得 2，i=7 时 2^3=8 得 3）。


与其他指标的区别：
    - Precision@K / Recall@K 不区分排名：相关来源排第 1 还是第 K，得分一样；
    - NDCG 引入 log2(i+1) 折损因子：排得越靠后贡献越小，排第 1 位贡献最大；
    - 分母 IDCG 是「天花板」：全部相关来源都排最前时 NDCG = 1.0。
"""

import math
from dataclasses import dataclass


@dataclass
class RetrievalCase:
    """一条检索评估样本。"""
    question: str  # 用户问题文本
    retrieved_sources: tuple[str, ...]  # 检索结果（按排名顺序排列，评估时截取前 K 条）
    relevant_sources: frozenset[str]  # 相关来源（人工标注的标准答案，无序），ID 级标准答案，供 Recall@K 等指标使用


def ndcg_at_k(case: RetrievalCase, k: int) -> float:
    """计算前K个结果的NDCG@K指标。"""

    if k <= 0:
        raise ValueError(f"k 必须为正整数，当前值为 {k}")

    # 1. 给前 K 个检索结果逐条打分，最终生成一个得分列表
    # 按原本顺序检查：首次命中相关来源记 1.0，未命中或重复出现记 0.0（例如 [0.0, 0.0, 1.0]）
    actual_scores = [
        1.0 if source in case.relevant_sources else 0.0
        for source in case.retrieved_sources[:k]
    ]

    # 2. 计算实际得分 DCG@K
    # 每一位的相关性得分除以 log2(排名+1)：排名越靠后，折损越严重
    dcg = sum(
        score / math.log2(rank + 1)
        for rank, score in enumerate(actual_scores, start=1)
    )

    # 3. 计算理想得分 IDCG@K （高分全放最前，不足 K 位补 0）
    # 理想情况：min(相关来源数, K) 条结果全部为 1.0，排在最前面
    ideal_relevant_count = min(len(case.relevant_sources), k)
    ideal_scores = [1.0] * ideal_relevant_count + [0.0] * (k - ideal_relevant_count)
    # 用与 DCG 相同的折损公式计算「天花板」得分
    idcg = sum(
        score / math.log2(rank + 1)
        for rank, score in enumerate(ideal_scores, start=1)
    )

    # 4. 归一化计算 NDCG@K；IDCG 为 0（无相关来源）时返回 0.0，避免除零
    # NDCG@K = DCG@K / IDCG@K
    ndcg = dcg / idcg if idcg > 0 else 0.0
    return ndcg


def print_case(case: RetrievalCase, k: int) -> None:
    """打印单条样本的完整指标和逐步说明。"""
    # 提取前K个结果
    top_k = case.retrieved_sources[:k]

    # 前K个结果中命中相关来源的数量（命中项列表）
    relevant_in_top_k = [source for source in top_k if source in case.relevant_sources]

    # 第一个相关来源的排名（enumerate 从 1 开始计排名；找不到则保持 None）
    first_relevant_rank = None
    for rank, source in enumerate(top_k, start=1):
        if source in case.relevant_sources:
            first_relevant_rank = rank
            break

    # 逐步打印样本信息，便于人工核对指标计算过程
    print(f"问题：{case.question}")
    print(f"返回顺序：{list(top_k)}")
    print(f"相关来源：{sorted(case.relevant_sources)}")
    print(f"前{k}条命中相关来源：{relevant_in_top_k}")
    print(f"第一个相关来源排名：{first_relevant_rank if first_relevant_rank else '无'}")
    print(f"NDCG@{k}={ndcg_at_k(case, k):.4f}")
    print()


if __name__ == "__main__":
    # 构造 4 条评估样本：模拟「电商售后 FAQ 知识库」的检索场景
    cases = [
        # 样本1：检索结果前两条全命中「售后规则.md」和「退款规则.md」 -> Hit@3 = 1
        RetrievalCase(
            question="退货条件和退款时间是什么？",
            retrieved_sources=("售后规则.md", "退款规则.md", "物流规则.md"),
            relevant_sources=frozenset({"售后规则.md", "退款规则.md"}),
        ),
        # 样本2：第 1 位命中唯一相关来源「物流规则.md」 -> Hit@3 = 1
        RetrievalCase(
            question="订单物流轨迹怎么查？",
            retrieved_sources=("物流规则.md", "发票规则.md", "售后规则.md"),
            relevant_sources=frozenset({"物流规则.md"}),
        ),
        # 样本3：相关来源只有「退款规则.md」在第 3 位，仍在 K 内 -> Hit@3 = 1
        RetrievalCase(
            question="退货后退款多久到账？",
            retrieved_sources=("发票规则.md", "物流规则.md", "退款规则.md"),
            relevant_sources=frozenset({"售后规则.md", "退款规则.md"}),
        ),
        # 样本4：唯一相关来源「积分规则.md」不在返回结果中 -> Hit@3 = 0
        RetrievalCase(
            question="会员积分什么时候清零？",
            retrieved_sources=("发票规则.md", "物流规则.md", "售后规则.md"),
            relevant_sources=frozenset({"积分规则.md"}),
        )
    ]

    k = 3  # 只评估检索结果的前 3 条
    metric_names = ("ndcg",)
    # 各指标的累加器，最后除以样本数得到平均分
    totals = {name: 0.0 for name in metric_names}

    print(f"评估样本数：{len(cases)}，K={k}")
    print("=" * 72)
    for case in cases:
        print_case(case, k)
        totals["ndcg"] += ndcg_at_k(case, k)  # 累加每条样本的 NDCG 得分

    count = len(cases)
    print("=" * 72)
    # 汇总：平均 NDCG@K = 所有样本 NDCG 之和 / 样本数
    print(
        f"汇总："
        f"NDCG@{k}={totals['ndcg'] / count:.4f}"
    )
