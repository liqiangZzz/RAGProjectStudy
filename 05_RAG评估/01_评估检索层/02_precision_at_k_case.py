"""Precision@K（精确率）评估示例。

指标含义：
    前 K 个检索结果中，相关来源所占的比例，衡量「检索回来的结果有多干净」。

公式：
    Precision@K = |前 K 个结果 ∩ 相关来源| / K

与 Hit@K 的区别：
    - Hit@K 只问「有没有命中」（0 或 1），Precision@K 问「命中了几个」；
    - 例如前 3 条中命中 2 条：Hit@3 = 1，而 Precision@3 = 2/3 ≈ 0.6667。

特点：
    - 取值范围 [0, 1]，越大说明前 K 条中无关结果越少；
    - 适合评估「用户看到的头部结果质量」，对垃圾结果敏感。
"""

from dataclasses import dataclass


@dataclass
class RetrievalCase:
    """一条检索评估样本。"""
    question: str  # 用户问题文本
    retrieved_sources: tuple[str, ...]  # 检索结果（按排名顺序排列，评估时截取前 K 条）
    relevant_sources: frozenset[str]  # 相关来源（人工标注的标准答案，无序）


def precision_at_k(case: RetrievalCase, k: int) -> float:
    """前K个结果中，相关来源的比例。"""
    if k <= 0:
        raise ValueError(f"k 必须为正整数，当前值为 {k}")
    # 提取前K个结果

    # 截取前 K 个结果，按集合交集计算命中数量，严格对应公式 |前 K 个结果 ∩ 相关来源| / K
    top_k = case.retrieved_sources[:k]
    hits = len(set(top_k) & case.relevant_sources)
    return hits / k


def print_case(case: RetrievalCase, k: int) -> None:
    """打印单条样本的完整指标和逐步说明。"""
    # 提取前K个结果
    top_k = case.retrieved_sources[:k]

    # 提取前K个结果中相关的来源的列表（命中项）
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
    print(f"precision@{k}={precision_at_k(case, k)}")
    print()


if __name__ == '__main__':
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
    metric_names = ("precision",)
    # 各指标的累加器，最后除以样本数得到平均分
    totals = {name: 0.0 for name in metric_names}

    print(f"评估样本数：{len(cases)}，K={k}")
    print("=" * 72)
    for case in cases:
        print_case(case, k)
        totals["precision"] += precision_at_k(case, k)  # 累加每条样本的精确率

    count = len(cases)
    print("=" * 72)
    # 汇总：平均 Precision@K = 所有样本精确率之和 / 样本数
    print(f"Precision@{k}={totals['precision'] / count:.4f}")
