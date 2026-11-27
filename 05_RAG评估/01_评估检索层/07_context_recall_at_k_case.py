"""Context Recall@K（上下文召回率，RAGAS 口径）示例。

指标定义：
    衡量标准参考答案（reference）所包含的关键事实信息，有多少比例能够被检索
    返回的前 K 条上下文切片（Chunk）所支撑/覆盖。

RAGAS 真实输入与判定机制：
    - 输入字段：user_input（问题）、reference（参考答案）、retrieved_contexts（有序切片列表）
    - 判定粒度：Claim（事实声明）级，而非文档 ID 级。
      1. 将 reference 拆解为若干条独立的原子事实声明（reference_claims）；
      2. 由 LLM Judge 判断每条声明是否能从检索到的切片文本中推断得出（归因）；
      3. 计算公式：Context Recall@K = 被前 K 条切片支撑的声明数 / 总声明数。

与 03 / 06 的分
    - 对比 03_recall_at_k：03 只看文档来源找没找全；07 把标准答案拆成原子事实，看关键信息覆盖了多少；
    - 配合 06_context_precision：06 管「有用片段排得好不好」，07 管「关键事实漏没漏（防幻觉）」。

说明：真实 RAGAS 会把前 K 条切片拼接后交给 LLM 整体判定；本示例不接大模型 API，
    直接用样本中的 llm_supported_claims 模拟每条切片能支撑的声明（取并集近似模拟）。
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievedChunk:
    """单条检索返回的上下文切片（纯粹模拟检索器输出，不含判定标签）。"""

    source: str  # 来源文件名（仅用于溯源打印，不参与指标计算）
    content: str  # 切片实际文本内容


@dataclass(frozen=True)
class ContextRecallCase:
    """一条对标 RAGAS SingleTurnSample 的 Context Recall 评估样本。"""

    user_input: str  # 用户问题
    reference: str  # 标准参考答案文本
    reference_claims: tuple[str, ...]  # 参考答案拆解出的独立事实声明（总声明）
    retrieved_contexts: tuple[RetrievedChunk, ...]  # 按检索排序返回的切片列表
    llm_supported_claims: tuple[frozenset[str], ...]  # 模拟 LLM Judge 事后判定：每条切片支撑了哪些声明

    def __post_init__(self) -> None:
        if len(self.retrieved_contexts) != len(self.llm_supported_claims):
            raise ValueError(
                f"retrieved_contexts ({len(self.retrieved_contexts)}) 与 "
                f"llm_supported_claims ({len(self.llm_supported_claims)}) 长度必须一致"
            )


def context_recall_at_k(case: ContextRecallCase, k: int) -> float:
    """按 RAGAS 声明（Claim）级口径计算前 K 个切片的 Context Recall@K。"""
    if k <= 0:
        return 0.0

    # 参考答案声明集合（去重）
    reference_claim_set = set(case.reference_claims)
    if not reference_claim_set:
        return 0.0

    # 汇总前 K 条切片共同支撑的声明（取并集去重，同一声明被多条切片支撑只计 1 次）
    supported_claims: set[str] = set()
    for claims in case.llm_supported_claims[:k]:
        # 把一个可迭代对象（如列表、元组、另一个集合）里的元素批量加进去。并且，如果遇到重复的元素，会自动忽略（去重）。
        supported_claims.update(claims)

    # 仅统计属于参考答案声明集合的条目，计算覆盖比例
    matched_claims = supported_claims & reference_claim_set
    return len(matched_claims) / len(reference_claim_set)


def print_case(case: ContextRecallCase, k: int) -> float:
    """打印单条样本的切片覆盖明细，并返回该样本的 Context Recall@K 得分。"""
    # 先计算得分（顺带完成 k <= 0 校验，避免后续切片出错）
    score = context_recall_at_k(case, k)

    top_k_chunks = case.retrieved_contexts[:k]
    top_k_supported = case.llm_supported_claims[:k]

    supported_claims: set[str] = set()
    for claims in top_k_supported:
        supported_claims.update(claims)

    matched_claims = [
        claim for claim in case.reference_claims if claim in supported_claims
    ]

    print(f"问题：{case.user_input}")
    print(f"参考答案：{case.reference}")
    print(f"参考答案声明（共{len(case.reference_claims)}条）：{list(case.reference_claims)}")
    print(f"前{k}条检索切片及 LLM Judge 判定：")
    for rank, (chunk, chunk_claims) in enumerate(zip(top_k_chunks, top_k_supported), start=1):
        hits = [c for c in case.reference_claims if c in chunk_claims]
        print(f"  [{rank}] 来源={chunk.source} | 支撑声明={hits} | 内容={chunk.content}")
    print(f"前{k}条累计支撑（{len(matched_claims)}/{len(case.reference_claims)}）：{matched_claims}")
    print(f"Context Recall@{k}={score:.4f}")
    print()

    return score


if __name__ == "__main__":
    cases = [
        # 样本1：前 2 条切片合力覆盖全部 2 条声明 -> 2/2 = 1.0000
        ContextRecallCase(
            user_input="退货条件和退款时间是什么？",
            reference="签收7天内保持商品完好可无理由退货；退货验收通过后3个工作日内原路退款。",
            reference_claims=(
                "7天无理由退货需保持商品完好",
                "退款将在验收通过后3个工作日内原路退回",
            ),
            retrieved_contexts=(
                RetrievedChunk(
                    source="售后规则.md",
                    content="支持7天无理由退货，需保证商品吊牌完整、未使用且不影响二次销售。",
                ),
                RetrievedChunk(
                    source="退款规则.md",
                    content="仓库确认退货无误后，退款将于3个工作日内原路退回原支付账户。",
                ),
                RetrievedChunk(
                    source="物流规则.md",
                    content="偏远地区默认使用顺丰陆运，预计3至5天送达。",
                ),
            ),
            llm_supported_claims=(
                frozenset({"7天无理由退货需保持商品完好"}),
                frozenset({"退款将在验收通过后3个工作日内原路退回"}),
                frozenset(),
            ),
        ),
        # 样本2：相关文档的切片只包含一半信息（只支撑 1/2 声明）-> 1/2 = 0.5000
        ContextRecallCase(
            user_input="订单物流轨迹怎么查？",
            reference="可在App订单详情页点击「查看物流」查看实时轨迹，发货后也会发送含物流单号的短信。",
            reference_claims=(
                "可在App订单详情页点击查看物流查看实时轨迹",
                "发货后系统会发送含物流单号的短信通知",
            ),
            retrieved_contexts=(
                RetrievedChunk(
                    source="物流规则.md",
                    content="打开App进入「我的订单-订单详情」，点击「查看物流」即可查看实时轨迹。",
                ),
                RetrievedChunk(
                    source="发票规则.md",
                    content="个人电子发票开具后不支持修改抬头。",
                ),
                RetrievedChunk(
                    source="售后规则.md",
                    content="生鲜水果类商品不支持7天无理由退货。",
                ),
            ),
            llm_supported_claims=(
                frozenset({"可在App订单详情页点击查看物流查看实时轨迹"}),
                frozenset(),
                frozenset(),
            ),
        ),
        # 样本3：第 2、3 条都来自「退款规则.md」，第 2 条切片无用，第 3 条切片独自支撑 3/4 声明 -> 0.7500
        ContextRecallCase(
            user_input="退货后退款多久到账？",
            reference="需先确认商品符合验收标准；审核通过后1个工作日内发起退款，微信支付宝实时到账，银行卡需1至3个工作日。",
            reference_claims=(
                "需先确认商品符合退货验收标准",
                "审核通过后1个工作日内发起退款",
                "微信或支付宝付款通常实时到账",
                "银行卡付款需1至3个工作日到账",
            ),
            retrieved_contexts=(
                RetrievedChunk(
                    source="售后规则.md",
                    content="7天无理由退货需买家自行承担寄回运费（质量问题除外）。",
                ),
                RetrievedChunk(
                    source="退款规则.md",
                    content="订单中使用的满减优惠券在退款时不予折现。",
                ),
                RetrievedChunk(
                    source="退款规则.md",
                    content="退货审核通过后1个工作日内发起退款，微信支付宝实时到账，银行卡1至3个工作日到账。",
                ),
            ),
            llm_supported_claims=(
                frozenset(),
                frozenset(),
                frozenset({
                    "审核通过后1个工作日内发起退款",
                    "微信或支付宝付款通常实时到账",
                    "银行卡付款需1至3个工作日到账",
                }),
            ),
        ),
        # 样本4：前 3 条切片均无法支撑参考答案中的任何声明 -> 0/2 = 0.0000
        ContextRecallCase(
            user_input="电子发票发到哪里？",
            reference="电子发票默认发送至下单预留邮箱，也可在订单详情页下载PDF版。",
            reference_claims=(
                "电子发票默认发送至下单预留邮箱",
                "也可在订单详情页下载PDF版电子发票",
            ),
            retrieved_contexts=(
                RetrievedChunk(
                    source="售后规则.md",
                    content="定制类商品确认生产后不支持无理由退换。",
                ),
                RetrievedChunk(
                    source="物流规则.md",
                    content="已发货订单如需改址请联系在线客服拦截。",
                ),
                RetrievedChunk(
                    source="退款规则.md",
                    content="银行卡退款受发卡行清算周期影响，节假日可能顺延。",
                ),
            ),
            llm_supported_claims=(
                frozenset(),
                frozenset(),
                frozenset(),
            ),
        ),
    ]

    if not cases:
        raise ValueError("评估样本列表不能为空")

    k = 3
    total = 0.0

    print(f"评估样本数：{len(cases)}，K={k}")
    print("=" * 72)
    for case in cases:
        total += print_case(case, k)

    print("=" * 72)
    print(f"汇总：Context Recall@{k}={total / len(cases):.4f}")