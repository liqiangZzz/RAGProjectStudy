"""
将PDF表格按“表头为单元格值”转换成检索文本
按页面视觉顺序（从上到下）输出文本块、表格、图片
"""
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Any

import pymupdf
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document

# ============================================================
# 配置路径
# ============================================================
FILE_PATH = Path(
    r"/Users/Python/project/project-learn/python-code/RAGProjectStudy/02_LangChain RAG构建流程/data/售后手册.pdf")
IMAGE_OUTPUT_DIR = Path(
    r"/Users/Python/project/project-learn/python-code/RAGProjectStudy/02_LangChain RAG构建流程/data/extracted_pdf_images")


# ============================================================
# 辅助函数：清理单元格文本
# ============================================================
def normalize_cell(cell: str | None) -> str:
    """清理单元格中的换行和多余空白，将连续空白替换为单个空格。"""
    # 如果 cell 是 None，转为空字符串；否则转为字符串
    # re.sub(r"\s+", " ", ...) 将所有连续空白（空格、换行、制表符）替换为单个空格
    # .strip() 去除首尾空白
    return re.sub(r"\s+", " ", str(cell or "")).strip()


# ============================================================
# 辅助函数：将表格行转换为自然语言陈述句
# ============================================================
def table_rows_to_text(rows: list[list[str | None]]) -> list[str]:
    """
    使用首行作为表头，将每条数据行转换成"字段为值"的完整陈述句。
    例如：表头 ["商品", "单价"]，数据 ["耳机", 299] → "商品为耳机，单价为299。"
    """

    # 1. 如果少于2行（无表头或无数据），直接返回空列表
    if len(rows) < 2:
        return []

    # 2. 提取并清理表头
    headers = [normalize_cell(cell) for cell in rows[0]]
    sentences: list[str] = []  # 初始化空列表，用于收集最终生成的每一行陈述句

    # 3. 遍历数据行，从第二行开始（跳过表头行，直接处理数据行）
    for row in rows[1:]:
        # 清理当前每个单元格的内容
        values = [normalize_cell(cell) for cell in row]

        # 用于存储当前行中，拼接好的一个个短句（如 "商品为耳机"）
        fields: list[str] = []

        # 遍历表头（enumerate 会同时返回列索引 index 和 表头名称 header）
        for index, header in enumerate(headers):
            # 如果表头为空，跳过该列（无法形成有意义的短语）
            if not header:
                continue

            # 防御性编程：如果当前数据行的列数比表头少（比如表格最后一行缺了数据），
            # 索引可能会越界，所以加上 `if index < len(values)` 防止程序崩溃。
            value = values[index] if index < len(values) else ""

            # 拼接短语，空值兜底显示为“未填写”
            fields.append(f"{header}为{value or '未填写'}")

        # 4.如果当前行有有效字段，用中文逗号连接，末尾加句号
        if fields:
            sentences.append("，".join(fields) + "。")

    return sentences


# ============================================================
# 定义页面元素的数据结构，用于按Y坐标排序
# ============================================================
@dataclass
class PageElement:
    """页面元素：文本块、表格或图片，统一记录其Y坐标和内容。"""
    y: float  # PDF 坐标系原点在左上角，Y值越小越靠上。用于最后统一排序。
    element_type: str  # 元素类型：text、table、image
    content: str  # 提取出的文本内容（或图片的 Markdown 引用）
    extra: dict[str, Any] | None = None  # 附加信息（如提取出的图片绝对路径）¬


# ============================================================
# 核心加载器类
# ============================================================
class RichPDFLoader(BaseLoader):
    """
    自定义 PDF 加载器：按视觉顺序（从上到下）提取正文、表格和图片引用，组装成 Document。
    """

    def __init__(self, file_path: Path, image_output_dir: Path) -> None:
        """初始化加载器：保存PDF路径和图片输出目录。"""
        self.file_path = file_path
        self.image_output_dir = image_output_dir

    def lazy_load(self) -> Iterator[Document]:
        """生成器：逐页处理 PDF。使用 yield 流式返回，节省内存。适合大文件流式处理。"""

        # 创建图片输出目录（如果不存在则递归创建）
        self.image_output_dir.mkdir(parents=True, exist_ok=True)

        # 使用 PyMuPDF 读取 PDF文件
        pdf = pymupdf.open(self.file_path)

        try:
            # 遍历每一页
            for page_index, page in enumerate(pdf):
                # 存储当前页的所有元素（文本块、表格、图片）
                elements: list[PageElement] = []

                # ============================================================
                # 第一步：识别表格，记录表格内容和表格区域
                # 必须先处理表格，确保表格内容在文本块之前，否则后续处理文本块时会错误识别为表格内容。
                # ============================================================

                # PyMuPDF 提供的表格检测方法，会分析页面上的线条、网格和文本布局，自动识别出页面中哪些区域属于“表格”，返回值是一个 TableFinder 对象
                table_result = page.find_tables()

                # TableFinder 对象的一个属性，返回一个列表（或元组），包含该页面上所有检测到的 Table 对象（即每一个独立的表格）
                # Table 对象的 bbox 属性，边界框（Bounding Box）。它是一个元组 (x0, y0, x1, y1)：
                # - x0：矩形左边缘的 X 坐标
                # - y0：矩形上边缘的 Y 坐标
                # - x1：矩形右边缘的 X 坐标
                # - y1：矩形下边缘的 Y 坐标
                # 坐标系原点在页面左上角，Y 轴向下为正。
                #
                # PyMuPDF 的 Rect 类构造函数，接收 (x0, y0, x1, y1) 元组，生成一个 Rect 对象。
                # Rect 对象提供了丰富的几何方法，如 .intersects()（检测两个矩形是否相交）、.contains()（检测是否包含某个点）等。
                # 最终得到的table_rects列表，包含该页面上所有表格的 Rect 对象。
                # 这里得到 table_rects 是方便后续判断文本是否存在于表格中，后续用到该 table_rects 列表
                table_rects = [pymupdf.Rect(table.bbox) for table in table_result.tables]

                # 提取内容并转换为陈述句，有几个表格就循环几次
                for table_index, table in enumerate(table_result.tables, start=1):
                    rows = table.extract()  # 提取原始二维列表
                    row_texts = table_rows_to_text(rows)  # 转换为自然语言文本

                    if row_texts:
                        # 构造表格的文本描述，加入标题
                        content = f"## 表格{table_index}\n" + "\n".join(row_texts)
                        table_y = table.bbox[1]  # 获取表格上边缘的 Y 坐标（PDF 坐标系原点在左上角，Y 值越小越靠上）
                        elements.append(
                            PageElement(
                                y=table_y,  # 表格Y坐标（越小越靠上）
                                element_type="table",
                                content=content,
                            )
                        )

                # ============================================================
                # 第二步：提取图片，保存为独立文件，并记录位置
                # ============================================================
                image_paths: list[str] = []  # 存储所有图片的绝对路径
                seen_xrefs: set[int] = set()  # 存储已处理的图片交叉引用编号，避免重复处理

                # page.get_images(full=True):遍历当前页面内容流中引用的所有图片，返回一个列表。
                # full=True: 控制返回信息的详细程度，设置为True时，get_images() 返回完整的图片元数据
                # image_info 每条信息包含 (xref, ...)等信息，xref是图片在 PDF 文件中的交叉引用编号（Cross-Reference ID），这是提取图片实际数据时所需的唯一标识符。
                for image_info in page.get_images(full=True):

                    # 图片的交叉引用编号
                    xref = image_info[0]
                    # 去重：同一张图片可能被多次引用
                    if xref in seen_xrefs:
                        continue
                    seen_xrefs.add(xref)

                    # extract_image(xref)：从 PDF 文件中提取指定图片的原始二进制数据及其元信息。返回 {"image": bytes, "ext": "png"}
                    image_data = pdf.extract_image(xref)

                    # 生成图片文件名：原文件名  _page_页码_image_序号.扩展名
                    image_number = len(image_paths) + 1
                    image_name = f"{self.file_path.stem}_page_{page_index + 1}_image_{image_number}.{image_data['ext']}"

                    # 等价于：Path(os.path.join(self.image_output_dir, image_name))
                    # 拼接图片路径，“/”表示Path中路径拼接符
                    image_path = self.image_output_dir / image_name

                    # 将图片二进制数据写入文件
                    image_path.write_bytes(image_data["image"])  # 向路径中写入图片二进制数据
                    image_paths.append(str(image_path.resolve()))  # 将图片路径添加到列表中

                    # page.get_image_rects(xref) 返回一个 Rect 对象的列表，精确定位每张图片在页面上的所有显示位置。
                    # 因为同一张图片可能在页面上出现多次，所以返回的是列表。
                    # 每个 Rect 对象内部包含边界坐标[x0，y0，x1，y1]
                    # - x0：矩形左边缘的 X 坐标
                    # - y0：矩形上边缘的 Y 坐标
                    # - x1：矩形右边缘的 X 坐标
                    # - y1：矩形下边缘的 Y 坐标
                    image_rects = page.get_image_rects(xref)
                    if image_rects:
                        # 取第一个矩形的位置（通常一张图片只出现一次）
                        img_y = image_rects[0].y0  # 正常情况取首个位置的 Y 坐标
                    else:
                        # 如果无法获取坐标，使用一个默认值（放在页面底部）
                        # page.rect：返回一个 Rect（矩形）对象，表示整个页面的边界框
                        # page.rect.height：返回矩形的高度（即 y1 - y0），由于页面矩形的 y0 通常为 0，所以 page.rect.height 直接等于页面底部的 Y 坐标值。
                        img_y = page.rect.height

                    # 构造图片的文本描述（Markdown格式引用）
                    image_content = f"![页面图片{image_number}]({Path(image_path).name})"
                    elements.append(
                        PageElement(
                            y=img_y,  # 图片Y坐标（越小越靠上）
                            element_type="image",  # 图片元素类型
                            content=image_content,  # 图片内容（Markdown格式引用）
                            extra={"path": str(image_path.resolve())},  # 图片绝对路径
                        )
                    )

                # ============================================================
                # 第三步：提取普通文本块（必须最后处理，执行双重过滤）（非表格、非图片区域）
                # ============================================================

                # page.get_text("blocks") 返回页面上的所有文本块
                # 每个块是一个元组: (x0, y0, x1, y1, text, block_no, block_type)
                # -x0: 文本块左边缘的 X 坐标
                # -y0：文本块上边缘的 Y 坐标
                # -x1：文本块右边缘的 X 坐标
                # -y1：文本块下边缘的 Y 坐标
                # -text：文本块的文本内容
                # -block_no：文本块的唯一标识符，块在页面上的编号（内部序号）
                # block_type：块类型标志：0 表示文本块（Text Block），1 表示图片块（Image Block）
                for block in page.get_text("blocks"):

                    # 过滤 1：跳过图片块（block_type=1）
                    if block[6] != 0:
                        continue  # 跳过非文本块（如图片块）

                    # 过滤 2：跳过表格区域内的文本块（防止表格内容被重复提取）
                    # PyMuPDF 的 Rect 类构造函数，接收 (x0, y0, x1, y1) 元组，生成一个 Rect 对象。
                    # Rect 对象提供了丰富的几何方法，如 .intersects()（检测两个矩形是否相交）、.contains()（检测是否包含某个点）等。
                    block_rect = pymupdf.Rect(block[:4])  # 文本块的矩形区域

                    # 判断文本块是否与表格区域相交。若在表格内则跳过（防止内容重复提取）
                    if any(block_rect.intersects(table_rect) for table_rect in table_rects):
                        continue  # 跳过表格中的文本块

                    # 提取文本块内容
                    block_text = str(block[4]).strip()
                    if not block_text:
                        continue

                    # 获取文本块的Y坐标（上边框的Y值，越小越靠上）
                    block_y = block_rect.y0

                    # 将文本块作为文本元素加入列表
                    elements.append(
                        PageElement(
                            y=block_y,  # 文本块Y坐标（越小越靠上）
                            element_type="text",  # 文本块元素类型
                            content=block_text,  # 文本块内容
                        )
                    )

                # ============================================================
                # 第四步：按Y坐标排序，保持视觉顺序（从上到下）
                # ============================================================
                elements.sort(key=lambda e: e.y)

                # ============================================================
                # 第五步：组装并生成 Document
                # ============================================================
                content_parts: list[str] = []
                # 当前页中按照y坐标排序好的结果，拼接页面内容
                for elem in elements:
                    content_parts.append(elem.content)

                # ============================================================
                # 第六步：创建Document，包含正文和元数据
                # ============================================================

                # 收集所有图片路径（用于元数据）
                all_image_paths = [
                    elem.extra["path"] for elem in elements
                    if elem.element_type == "image" and elem.extra
                ]

                yield Document(
                    page_content="\n\n".join(content_parts),
                    metadata={
                        "source": str(self.file_path.resolve()),  # 源文件绝对路径
                        "page": page_index,  # 页码索引（从0开始）
                        "table_count": len(table_result.tables),  # 识别到的表格数量
                        "image_count": len(all_image_paths),  # 识别到的图片数量
                        "images": all_image_paths,  # 所有图片路径（用于元数据）
                    },
                )

        finally:
            # 确保PDF文件被正确关闭，释放资源
            pdf.close()


# ============================================================
# 主流程：执行加载并打印结果
# ============================================================
"""创建加载器，加载PDF，打印每页的元数据和正文内容。"""
loader = RichPDFLoader(
    file_path=FILE_PATH,
    image_output_dir=IMAGE_OUTPUT_DIR,
)

# load() 内部调用 lazy_load() 并收集所有Document
documents = loader.load()

print("Document数量：", len(documents))

for document in documents:
    print("页码索引：", document.metadata["page"])
    print("识别表格数：", document.metadata["table_count"])
    print("识别图片数：", document.metadata["image_count"])
    print("图片文件：", [Path(path).name for path in document.metadata["images"]])

    print("\nDocument正文：")
    print(document.page_content)
