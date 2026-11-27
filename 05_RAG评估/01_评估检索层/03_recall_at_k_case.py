"""Recall@K（召回率）评估示例。

指标含义：
    全部相关来源中，被前 K 个检索结果覆盖的比例，衡量「该找的有没有找全」。

公式：
    Recall@K = |前 K 个结果 ∩ 相关来源| / |相关来源|

与 Precision@K 的区别（视角相反）：
    - Precision：分子分母都以「检索结果」为出发点 —— 找回来的里面有多少是对的；
    - Recall：以「相关来源」为出发点 —— 该找的里面找回来了多少。

特点：
    - 取值范围 [0, 1]，越大说明漏掉的相关资料越少；
    - 对 RAG 很关键：漏掉关键文档（低召回）时，LLM 缺少生成答案的依据，
      答案大概率出错 —— 这就是「检索召回不足导致幻觉」的典型成因。
"""

from dataclasses import dataclass


@dataclass
class RetrievalCase:
    """一条检索评估样本。"""
    question: str  # 用户问题文本
    retrieved_sources: tuple[str, ...]  # 检索结果（按排名顺序排列，评估时截取前 K 条）
    relevant_sources: frozenset[str]  # 相关来源（人工标注的标准答案，无序）


def recall_at_k(case: RetrievalCase, k: int) -> float:
    """全部相关来源中，被前K个结果覆盖的比例。"""
    if k <= 0:
        raise ValueError(f"k 必须为正整数，当前值为 {k}")

    #  如果没有相关来源，召回率为0（避免除零错误）
    if not case.relevant_sources:
        return 0.0

    # 截取前 K 个结果，与相关来源做交集，即「找回来的相关文档」
    hit_sources = set(case.retrieved_sources[:k]) & (case.relevant_sources)

    # 计算召回率 = 命中相关来源数 / 全部相关来源数
    return len(hit_sources) / len(case.relevant_sources)


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
    print(f"Recall@{k}={recall_at_k(case, k):.4f}")
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
    metric_names = ("recall",)
    # 各指标的累加器，最后除以样本数得到平均分
    totals = {name: 0.0 for name in metric_names}

    print(f"评估样本数：{len(cases)}，K={k}")
    print("=" * 72)
    for case in cases:
        print_case(case, k)
        totals["recall"] += recall_at_k(case, k)  # 累加每条样本的召回率

    count = len(cases)
    print("=" * 72)
    # 汇总：平均 Recall@K = 所有样本召回率之和 / 样本数
    print(
        f"Recall@{k}={totals['recall'] / count:.4f}"
    )
