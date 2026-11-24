import re
from dataclasses import dataclass

import numpy as np
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.documents import Document

from env_utils import DASHSCOPE_API_KEY


@dataclass
class SemanticSplitAnalysis:
    """保存语义切分过程的中间结果"""
    sentences: list[str]          # 原始句子列表
    distances: list[float]        # 相邻句子之间的 Embedding 距离列表
    threshold: float              # 动态计算出的语义断点阈值
    break_after: list[int]        # 最终确认的语义断点索引列表


class EmbeddingSemanticTextSplitter:
    """基于 Embedding 距离的语义切分器。在话题发生转换的地方自动下刀。"""

    def __init__(
            self,
            embeddings,
            breakpoint_percentile: float = 75.0,
            min_sentences: int = 3,
            max_chunk_chars: int = 180
    ) -> None:
        self.embeddings = embeddings  # 用于计算Embedding距离的模型
        self.breakpoint_percentile = breakpoint_percentile  # 用于计算语义断点的距离分位数
        self.min_sentences = min_sentences                # 最小句子数限制（防止切得太碎）
        self.max_chunk_chars = max_chunk_chars            # 最大字符数限制（防止块太大超出模型上下文）

    def split_sentences(self, text: str) -> list[str]:
        """将长文本按照中文标点切分为独立的句子列表。"""

        # 去除所有空白字符（换行、空格、制表符），避免干扰正则匹配
        text = re.sub(r"\s+", "", text, )

        # 按中文句号、问号、感叹号和分号切分
        # 正则解释：".+?" 非贪婪匹配任意字符，(?:[。！？；]|$) 匹配中文句尾标点或字符串末尾
        sentences = re.findall(r".+?(?:[。！？；]|$)", text)

        # 过滤掉可能产生的空字符串
        return [sentence for sentence in sentences if sentence.strip()]

    def cosine_distance(self, vector_a: np.ndarray, vector_b: np.ndarray) -> float:
        """
        计算两个向量的余弦距离（1 - 余弦相似度）。
        距离越小，语义越接近；距离越大，语义跨度越大。
        """

        # 计算两个向量的 L2 范数（长度）的乘积
        # np.linalg.norm(v)：计算向量的 L2 范数（欧几里得长度），公式为 sqrt(sum(v_i²))
        denominator = (np.linalg.norm(vector_a) * np.linalg.norm(vector_b))

        # 异常兜底：若向量长度为空，直接视为完全无关
        if denominator == 0:
            return 1.0

        # 计算余弦相似度：(A·B) / (|A|*|B|)
        # np.dot(vector_a, vector_b)：两个向量的点积，公式为 Σ a_i · b_i
        similarity = float(np.dot(vector_a, vector_b) / denominator)

        # 修正浮点数精度误差，确保值域在 [-1, 1]
        similarity = max(-1.0, min(1.0, similarity))

        # 余弦距离 = 1- 余弦相似度
        return 1.0 - similarity

    def analyze(self, sentences: list[str]) -> SemanticSplitAnalysis:
        """
        根据句子列表，得到 SemanticSplitAnalysis 对象

        Args:
            sentences (list[str]): 输入的句子列表，每个句子是一个字符串。
        Returns:
            SemanticSplitAnalysis: 包含原始句子列表、相邻句子之间的Embedding距离列表、语义断点的阈值和语义断点索引列表的 SemanticSplitAnalysis 对象。
        """

        # 至少2个句子，才能计算语义断点
        if len(sentences) < 2:
            return SemanticSplitAnalysis(sentences, [], 0.0, [])

        # 1.计算所有句子的Embedding向量表示
        vectors = self.embeddings.embed_documents(sentences)
        arrays = [np.asarray(vector, dtype=np.float32) for vector in vectors]  # 转换为numpy数组，为了后续计算矩阵距离时速度更快，且避免类型报错。

        # 2. 计算相邻句子之间的余弦距离
        distances = [self.cosine_distance(a, b) for a, b in zip(arrays[:-1], arrays[1:])] # arrays[:-1] 是第一个到倒数第二个，arrays[1:] 是第二个到最后一个

        # ============================================================
        # 3. 动态计算语义切分的阈值（Threshold）（动态自适应）
        # 核心目的：取 75% 分位数，划出一条“门槛线”，找出距离最大的前 25% 的突变点。
        #
        # 为什么要用分位数？因为硬编码一个固定值（比如规定大于0.5就切）是不靠谱的。
        # 有的文章句子间距离普遍偏小，有的偏大。用 75% 分位数，能根据当前文档的实际情况，
        # 自动拉出一条警戒线，把语义最不连贯的地方（前25%）抓出来。
        #
        # 计算步骤（假设分位数 P = 75%）：
        #   1. 将距离列表 distances 从小到大排序，假设长度为 N。
        #   2. 计算目标位置索引：(N - 1) * (P / 100)。
        #   3. 取值：如果索引是整数，直接取；如果是小数（如 6.75），则取第 6 位和第 7 位进行线性插值。
        #
        # 插值公式通俗拆解（假设第 6 个值是 0.7，第 7 个值是 0.8，索引小数是 0.75）：
        #   1. (0.8 - 0.7)：用“大值减小值”，算出相邻两点之间的【总跨度】。
        #   2. 然后再 * 0.75：乘以进度比例，算出按进度需要走多长的【增量】。
        #   3. 0.7 + 增量：起点值加上增量，得到最终精确的插值结果（0.775）。

        # 举例：distances = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        #   - N = 10,P = 75%
        #   - 目标索引：(10 - 1) * 0.75 = 6.75 （卡在第 6 个值 0.7 和第 7 个值 0.8 之间）
        #   - 取第 6 个值（0.7）和第 7 个值（0.8）
        #   - 插值计算：0.7 + (0.8 - 0.7) * 0.75 = 0.775
        # 结果：threshold = 0.775。后续代码只需 `distance >= threshold`，就能精准筛出前 25% 的突变点。
        # ============================================================
        threshold = float(np.percentile(np.asarray(distances), self.breakpoint_percentile))

        # 4. 找出所有“语义突变”的候选断点（下标）
        # 用前面算出的门槛(threshold)去过滤距离列表，提取出符合条件的位置。
        # 例如距离列表=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]，threshold=0.775，这里将返回下标列表 [7, 8, 9]。
        # 推导过程：
        #   索引 0，值 0.1：0.1 >= 0.775 ❌（不成立）
        #   ...中间都不成立...
        #   索引 6，值 0.7：0.7 >= 0.775 ❌（不成立）
        #   索引 7，值 0.8：0.8 >= 0.775 ✅ （成立，记录索引 7）
        #   索引 8，值 0.9：0.9 >= 0.775 ✅ （成立，记录索引 8）
        #   索引 9，值 1.0：1.0 >= 0.775 ✅ （成立，记录索引 9）
        candidates = [index for index, distance in enumerate(distances) if distance >= threshold]

        # 5. 过滤候选断点：防止两个断点靠得太近，导致切出来的 Chunk 过短
        break_after = []  # 存储最终真正要“下刀”切分的句子索引位置
        previous_break = -1  # 用 -1 保证第一个候选点（如 index=0）的距离计算无误 (0 - (-1) = 1)

        # 遍历前面找出的所有“语义候选断点”
        for index in candidates:
            # 判断当前候选点距离上一次切分点，是否达到了最小句子数限制
            if index - previous_break >= self.min_sentences:
                break_after.append(index)
                previous_break = index  # 更新切分点，作为下一轮计算的起点

        # 返回 SemanticSplitAnalysis 对象
        return SemanticSplitAnalysis(
            sentences=sentences,
            distances=distances,
            threshold=threshold,
            break_after=break_after
        )

    def split_document(self, document: Document) -> tuple[list[Document], SemanticSplitAnalysis]:
        """
        执行文档切分主流程：先切句子，再算断点，最后智能组装成多个 Chunk。

        Args:
            document (Document): 输入的原始 LangChain Document 对象。

        Returns:
            tuple[list[Document], SemanticSplitAnalysis]:
                - 切分好的 Document 列表
                - 语义分析结果对象（包含距离、阈值、断点等，用于调试和追溯）
        """

        # 1. 文本切分：按中文标点，将长文档切分为独立的句子列表
        sentences = self.split_sentences(document.page_content)

        # 2. 语义分析：调用 analyze 方法，得到语义SplitAnalysis 对象，算出所有相邻句子的距离，并得出最终的语义断点索引（break_after）
        analysis = self.analyze(sentences)

        # 3.根据 SemanticSplitAnalysis 获取Document 列表

        # 3.1 智能组装 Chunk（双保险策略：语义断点优先，长度超限兜底）
        groups: list[list[str]] = []  # 最终句子分组列表
        current: list[str] = []  # 当前正在累积的句子组
        current_length = 0  # 当前组的总字符数

        for index, sentence in enumerate(analysis.sentences):
            # 将当前句子加入当前组，并累加字符数
            current.append(sentence)
            current_length += len(sentence)

            # --- 切断条件一：自然结束 ---
            # 如果当前是全文最后一句，无论如何都要在此处收尾
            is_last = (index == len(analysis.sentences) - 1)

            # --- 切断条件二：物理超长（前瞻性判断） ---
            # 逻辑：提前看看“下一句”的长度。如果加上下一句会超过 max_chunk_chars，就在这里强制截断。
            # 目的：防止 Chunk 突破上下文窗口限制，同时保证“句子完整性”（不会把一句话切两半）。
            next_length = (len(analysis.sentences[index + 1]) if not is_last else 0)
            exceeds_limit = (not is_last and current_length + next_length > self.max_chunk_chars)

            # --- 切断条件三：语义断点（软约束） ---
            # 如果当前句子的下标，正好在之前的语义分析 analyze 中被判定为“话题突变点”，那就强制封箱。保证一个块里只有同一个主题。
            semantic_break = index in analysis.break_after

            # 只要满足上述任意一个条件，立刻打包成组，并清空重置
            if is_last or exceeds_limit or semantic_break:
                groups.append(current)
                current = []
                current_length = 0

        # 4. 将句子组转换为 LangChain 的 Document 对象
        chunks: list[Document] = []
        sentence_cursor = 0  # 游标：记录当前 Chunk 在原句子列表中的起始位置

        for chunk_index, group in enumerate(groups):
            start = sentence_cursor # 当前chunk的起始句子索引
            sentence_cursor += len(group)  # 游标向前移动当前组的句子数量
            end = sentence_cursor - 1 # 当前chunk的结束句子索引

            # 创建新的Document
            chunks.append(
                Document(
                    # 将同一个chunk中的句子拼接起来
                    page_content="".join(group),
                    # 保留原Document的metadata，
                    # 同时追加语义切分信息
                    metadata={
                        **document.metadata,
                        "splitter": "embedding_semantic_splitter", # 记录切分器类型
                        "chunk_index": chunk_index,  # 当前块在整篇文档中的序号
                        "sentence_start": start,  # 【溯源】当前块起始于原文档的第几句
                        "sentence_end": end,    # 【溯源】当前块结束于原文档的第几句
                        "breakpoint_percentile": self.breakpoint_percentile, # 记录切分参数
                        "breakpoint_threshold": round(analysis.threshold, 6, ),
                        "max_chunk_chars": self.max_chunk_chars,
                    },
                )
            )

        return chunks, analysis


def main() -> None:
    # ========================================================
    # 第一步：初始化真实Embedding模型
    # ========================================================
    embedding_model = DashScopeEmbeddings(
        model="text-embedding-v1",
        dashscope_api_key=DASHSCOPE_API_KEY,
    )

    # ========================================================
    # 第二步：准备多个业务主题的售后文档
    # ========================================================
    source_document = Document(
        page_content=(
            "退货规则：收货后七日内可以申请无理由退货，"
            "商品、包装和配件需要保持完整。"
            "退货申请提交后，客服会在一个工作日内完成审核。"
            "审核通过后，用户需要将商品寄回指定仓库。"

            "退款规则：仓库验收通过后发起原路退款，"
            "通常三个工作日内到账。"
            "如果商品存在质量问题，"
            "退款申请需要附上相关照片。"
            "退款会按照原支付渠道原路退回。"

            "物流规则：订单发货后可以在订单详情查看物流轨迹。"
            "物流信息通常会在包裹揽收后更新。"
            "如果物流超过三天没有更新，"
            "需要联系人工客服查询。"

            "保修规则：耳机主机提供一年有限保修服务。"
            "人为损坏和进水造成的故障不属于免费保修范围。"
            "申请保修时需要提供订单号和故障说明。"
        ),
        metadata={"source": "售后规则.md"},
    )

    # ========================================================
    # 第三步：创建自定义语义Splitter
    # ========================================================
    splitter = EmbeddingSemanticTextSplitter(
                embeddings=embedding_model, # 语义模型
                breakpoint_percentile=75.0, # 断点百分比
                min_sentences=3, # 最小句子数量
                max_chunk_chars=180, # 最大chunk字符数
               )

    # 对整个Document执行一次切分,这里同时拿到切分结果和语义分析结果，
    # 方便打印距离和断点。
    chunks, analysis = splitter.split_document(source_document)

    print("句子数量：",len(analysis.sentences),)

    print("相邻句子语义距离：")
    for index, distance in enumerate(analysis.distances):
        print(f"句子{index} -> 句子{index + 1}:{distance:.6f}")

    print("断点阈值：",f"{analysis.threshold:.6f}",)
    print("语义断点索引列表：",analysis.break_after,)
    print("Document数量：",len(chunks),)

    for index, chunk in enumerate(chunks):
        print(f"Document{index}：{chunk.page_content}")
        print("metadata：",chunk.metadata,)


if __name__ == "__main__":
    main()