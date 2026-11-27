"""MRR（Mean Reciprocal Rank，平均倒数排名）评估示例。

指标含义：
    - RR（Reciprocal Rank）：针对单条查询，记前 K 条结果中首个相关来源的排序位次为 r，
      则 RR = 1 / r；若前 K 条内未命中任何相关来源，则记 RR = 0。
    - MRR（Mean Reciprocal Rank）：评估集上所有样本 RR 的算术均值，用于衡量检索系统
      将首个相关来源置于返回结果前列的能力。

公式：
    RR = 1 / rank(第一个相关来源)，前 K 内无命中则为 0
    MRR = Σ RR / 样本数（把所有样本各自的 RR 加起来求和，再以样本数归一化。）

与 Hit@K 的区别：
    - Hit@K 只区分「命中/未命中」，不管第几名命中；
    - RR 对排名敏感：排第 1 位得 1.0、第 2 位得 0.5、第 3 位得 0.3333；
    - 排名越靠前，用户越早看到答案，体验越好 —— 所以 MRR 比 Hit@K 更「挑剔」。
"""

from dataclasses import dataclass


@dataclass
class RetrievalCase:
    """一条检索评估样本。"""
    question: str  # 用户问题文本
    retrieved_sources: tuple[str, ...]  # 检索结果（按排名顺序排列，评估时截取前 K 条）
    relevant_sources: frozenset[str]  # 相关来源（人工标注的标准答案，无序）


def reciprocal_rank(case: RetrievalCase, k: int) -> float:
    """从检索结果中，计算第一个相关来源的排名倒数。"""

    if k <= 0:
        raise ValueError(f"k 必须为正整数，当前值为 {k}")

    # 遍历前 K 个结果，rank 从 1 开始计排名
    for rank, source in enumerate(case.retrieved_sources[:k], start=1):
        # 遇到第一个相关来源 -> 立即返回排名倒数，不再往后看（第一个命中即返回）
        if source in case.relevant_sources:
            return 1.0 / rank

    # 前 K 个结果中没有任何相关来源 -> 记 0 分
    return 0.0


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
    print(f"RR={reciprocal_rank(case, k):.4f}")
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
    metric_names = ("mrr",)
    # 各指标的累加器，最后除以样本数得到平均分
    totals = {name: 0.0 for name in metric_names}

    print(f"评估样本数：{len(cases)}，K={k}")
    print("=" * 72)
    for case in cases:
        print_case(case, k)
        totals["mrr"] += reciprocal_rank(case, k)  # 累加每条样本的倒数排名

    count = len(cases)
    print("=" * 72)
    # 汇总：MRR = 所有样本 RR 之和 / 样本数
    print(
        f"汇总："
        f"MRR={totals['mrr'] / count:.4f}，"
    )
