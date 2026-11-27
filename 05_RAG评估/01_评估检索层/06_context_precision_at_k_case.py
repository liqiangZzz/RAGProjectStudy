"""Context Precision@K（上下文精确率，RAGAS 口径）示例。

指标定义：
    评估检索返回的前 K 条切片（Chunk）中，真正有助于得出参考答案（reference）的
    片段「是否排在靠前位置」。有用切片越靠前、中间夹杂的无关内容越少，得分越高。

RAGAS 真实输入与判定机制：
    - 输入字段：user_input（问题）、reference（参考答案）、retrieved_contexts（有序切片列表）
    - 判定粒度：Chunk 文本级（非文档 ID 级）。LLM Judge 会逐一读取每个切片内容，
      判断其对生成 reference 是否有用，输出结构化判定 verdict ∈ {0, 1}。
    - 聚合公式（只看 Top-K 内部）：
        Context Precision@K = Σ(Precision@i × v_i) / Top-K 中有用切片总数
        其中 Precision@i = 截至第 i 位有用切片的累计数 / i
       （例如前 3 条判定为 [有用, 无用, 有用]：第 1 位得 1/1，第 3 位得 2/3 -> (1 + 2/3) / 2 = 0.8333）

与 02 / 07 的分工：
    - 对比 02_precision_at_k：02 只看文档来源对不对、不看顺序；06 看切片内容是否有助解答，且顺序越靠前分越高；
    - 配合 07_context_recall：06 只在 Top-K 内部算分（管「排得好不好」），漏召回由 07 负责兜底。

说明：
    真实 RAGAS 会调用 LLM 对照 reference 给每条切片打分；本示例不接大模型 API，
    直接用样本中的 llm_verdicts（1=有用，0=无用）模拟裁判结果。
"""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class RetrievedChunk:
    """单条检索返回的上下文切片（纯粹模拟检索器输出，不含判定标签）。"""

    source: str  # 来源文件名（仅用于溯源打印，不参与指标计算）
    content: str  # 切片实际文本内容（RAGAS 中 LLM Judge 真正读取并判定的对象）


@dataclass(frozen=True)
class ContextPrecisionCase:
    """一条对标 RAGAS SingleTurnSample 的 Context Precision 评估样本。"""

    user_input: str  # 用户问题
    reference: str  # 标准参考答案文本
    retrieved_contexts: tuple[RetrievedChunk, ...]  # 按检索排序返回的切片列表
    llm_verdicts: tuple[Literal[0, 1], ...]  # 模拟 LLM Judge 事后对每条切片的判定结果（1=有用，0=无用）

    def __post_init__(self) -> None:
        if len(self.retrieved_contexts) != len(self.llm_verdicts):
            raise ValueError(
                f"retrieved_contexts ({len(self.retrieved_contexts)}) 与 "
                f"llm_verdicts ({len(self.llm_verdicts)}) 长度必须一致"
            )

def context_precision_at_k(case: ContextPrecisionCase, k: int) -> float:
    """按 RAGAS 口径计算前 K 个切片的 Context Precision@K。

    计算示例（假设前 3 条切片依次为 [有用, 无用, 有用]）：
        - rank=1（有用）：累计 1 条 -> Precision@1 = 1/1 = 1.0000
        - rank=2（无用）：跳过，不累加，但占用第 2 位使后续分母变大
        - rank=3（有用）：累计 2 条 -> Precision@3 = 2/3 ≈ 0.6667
        - 最终取平均：(1.0000 + 0.6667) / 2 = 0.8333
    """

    if k <= 0:
        raise ValueError(f"k 必须为正整数，当前值为 {k}")

    top_k_verdicts = case.llm_verdicts[:k]

    # 1. 统计 Top-K 中被判定为有用的切片总数（作为分母）
    useful_count = sum(top_k_verdicts)
    if useful_count == 0:
        return 0.0

    # 2. 累加每个有用切片所在位置的前缀精确率 Precision@rank
    weighted_precision = 0.0
    running_useful = 0
    for rank, verdict in enumerate(top_k_verdicts, start=1):
        if verdict == 1:
            running_useful += 1
            # 当前位置前缀精确率 = 截至本位的有用切片数 / 当前排名
            weighted_precision += running_useful / rank

    # 3. 对所有有用位置的前缀精确率取平均
    return weighted_precision / useful_count


def print_case(case: ContextPrecisionCase, k: int) -> float:
    """打印单条样本的切片判定明细与 Context Precision@K。"""

    # 先计算得分（顺带完成 k <= 0 校验，避免后续切片出错）
    score = context_precision_at_k(case, k)

    top_k_chunks = case.retrieved_contexts[:k]
    top_k_verdicts = case.llm_verdicts[:k]

    print(f"问题：{case.user_input}")
    print(f"参考答案：{case.reference}")
    print(f"前{k}条检索切片及 LLM Judge 判定（verdict）：")
    for rank, (chunk, verdict) in enumerate(zip(top_k_chunks, top_k_verdicts), start=1):
        tag = "有用(1)" if verdict == 1 else "无用(0)"
        print(f"  [{rank}] {tag} | 来源={chunk.source} | 内容={chunk.content}")
    print(f"判定序列 v_i：{list(top_k_verdicts)} -> Context Precision@{k}={score:.4f}")
    print()
    return score


if __name__ == "__main__":
    cases = [
        # 样本1：第 1、3 位切片有用，第 2 位夹杂无关切片
        # -> Precision@1 = 1/1 = 1.0, Precision@3 = 2/3 -> (1.0 + 2/3) / 2 = 0.8333
        ContextPrecisionCase(
            user_input="退货条件和退款时间是什么？",
            reference="签收7天内保持商品完好可无理由退货；退货验收通过后3个工作日内原路退款。",
            retrieved_contexts=(
                RetrievedChunk(
                    source="售后规则.md",
                    content="支持7天无理由退货，需保证商品吊牌完整、未使用且不影响二次销售。",
                ),
                RetrievedChunk(
                    source="物流规则.md",
                    content="偏远地区默认使用顺丰陆运，预计3至5天送达。",
                ),
                RetrievedChunk(
                    source="退款规则.md",
                    content="仓库确认退货无误后，退款将于3个工作日内原路退回原支付账户。",
                ),
            ),
            llm_verdicts=(1, 0, 1),
        ),
        # 样本2：唯一有用切片排在第 1 位 -> Precision@1 = 1.0 -> 1.0000
        ContextPrecisionCase(
            user_input="订单物流轨迹怎么查？",
            reference="在App订单详情页点击「查看物流」即可查询实时配送轨迹。",
            retrieved_contexts=(
                RetrievedChunk(
                    source="物流规则.md",
                    content="打开App进入「我的订单-订单详情」，点击「查看物流」可实时查看快递轨迹。",
                ),
                RetrievedChunk(
                    source="发票规则.md",
                    content="个人电子发票开具后不支持修改抬头，如需重开请联系客服。",
                ),
                RetrievedChunk(
                    source="售后规则.md",
                    content="生鲜水果类商品不支持7天无理由退货。",
                ),
            ),
            llm_verdicts=(1, 0, 0),
        ),
        # 样本3（体现 Chunk 级 vs ID 级的关键差异）：
        # 第 2、3 位切片来自【同一份文档】退款规则.md，但第 2 位讲的是“优惠券退回”，与“到账时间”无关(0)；
        # 只有第 3 位讲“退款到账时效”(1) -> 得分 = (1/3) / 1 = 0.3333
        ContextPrecisionCase(
            user_input="退货后退款多久到账？",
            reference="退货验收后1个工作日内发起退款，微信或支付宝实时到账，银行卡需1至3个工作日。",
            retrieved_contexts=(
                RetrievedChunk(
                    source="售后规则.md",
                    content="7天无理由退货需由买家承担寄回运费（质量问题除外）。",
                ),
                RetrievedChunk(
                    source="退款规则.md",
                    content="订单中使用的满减优惠券在退款时不折现，满足条件将退回卡包。",
                ),
                RetrievedChunk(
                    source="退款规则.md",
                    content="退货验收通过后1个工作日内发起退款，微信支付宝实时到账，银行卡1至3个工作日到账。",
                ),
            ),
            llm_verdicts=(0, 0, 1),
        ),
        # 样本4：Top-K 全为无关切片 -> 0.0000
        ContextPrecisionCase(
            user_input="电子发票发到哪里？",
            reference="电子发票默认发送至下单预留邮箱，也可在订单详情页下载PDF。",
            retrieved_contexts=(
                RetrievedChunk(
                    source="售后规则.md",
                    content="定制类商品确认生产后不支持无理由退换。",
                ),
                RetrievedChunk(
                    source="物流规则.md",
                    content="已发货订单如需改址，请联系在线客服申请拦截。",
                ),
                RetrievedChunk(
                    source="退款规则.md",
                    content="银行卡退款受清算周期影响，节假日可能顺延。",
                ),
            ),
            llm_verdicts=(0, 0, 0),
        ),
    ]

    k = 3
    total = 0.0

    print(f"评估样本数：{len(cases)}，K={k}")
    print("=" * 72)
    for case in cases:
        total += print_case(case, k)

    print("=" * 72)
    print(f"汇总：Context Precision@{k}={total / len(cases):.4f}")