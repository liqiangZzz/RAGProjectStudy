"""
处理DOCX文件，按DOCX原始块顺序读取正文、表格和图片
段落、表格和图片按DOCX内部顺序写入Document
"""
import re
from pathlib import Path
from typing import Iterator

from docx import Document as WordDocument
from docx.document import Document as WordDocumentType
from docx.oxml import CT_P, CT_Tbl
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document

# ============================================================
# 配置路径
# ============================================================
FILE_PATH = Path(
    r"/Users/Python/project/project-learn/python-code/RAGProjectStudy/02_LangChain RAG构建流程/data/售后制度.docx")
IMAGE_OUTPUT_DIR = Path(
    r"/Users/Python/project/project-learn/python-code/RAGProjectStudy/02_LangChain RAG构建流程/data/extracted_docx_images")


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
# 按照DOCX内部顺序遍历段落和表格
# ============================================================
def iter_document_blocks(document: WordDocumentType) -> Iterator[Document]:
    """
    按照DOCX内部XML顺序逐个产生段落和表格对象。
    document.paragraphs 和 document.tables 各自独立，无法保留交叉顺序。
    直接遍历 body 的 XML 子节点，顺序即文档的真实视觉顺序。
    """

    # document.element.body 是文档主体的 XML 根节点 <w:body>
    # .iterchildren() 遍历 body 的所有直接子节点

    for child in document.element.body.iterchildren():
        # CT_P 是 <w:p>节点，表示一个段落
        if isinstance(child, CT_P):
            yield Paragraph(child, document)

        # CT_Tbl 是 <w:tbl>节点，表示一个表格
        elif isinstance(child, CT_Tbl):
            yield Table(child, document)

# ============================================================
# 第四步：提取Word表格的二维数据
# ============================================================
def extract_table_rows(table: Table) -> list[list[str]]:
    """提取Word表格中的二维行列数据，每个单元格的文本内容经过清理。"""
    return [[normalize_cell(cell.text) for cell in row.cells] for row in table.rows]



# ============================================================
# 第五步：提取段落中的嵌入图片
# ============================================================
def extract_paragraph_images(
        paragraph: Paragraph,
        document: WordDocumentType,
        image_output_dir: Path,
        file_stem: str,
        image_counter: int,
) -> tuple[list[str], int]:
    """
    提取段落中嵌入的图片。

    Word 中图片通过 <w:drawing> → <a:blip> 节点引用，
    <a:blip> 的 r:embed 属性保存关系 ID，
    通过关系 ID 可在 document.part.related_parts 中找到对应的图片部件。

    返回值：(当前段落提取出的图片路径列表, 更新后的图片计数器)
    """
    image_paths: list[str] = []

    # paragraph._p 是段落底层的 XML 节点 <w:p>
    # .xpath(".//a:blip") 查找所有 <a:blip> 节点元素，这是 Word 中图片的引用的关键节点
    # qn("r:embed") 构造带有命名空间的标签名，用于读取 r:embed 属性
    for blip in paragraph._p.xpath(".//a:blip"):
        # 读取 r:embed 属性值，即图片的关系 ID
        relation_id = blip.get(qn("r:embed"))
        # 没有关系 ID，跳过该图片
        if not relation_id:
            continue

        # 通过关系 Id 找到对应的图片部件
        image_part = document.part.related_parts[relation_id]

        # 从 MIME 类型中提取图片扩展名，如 "image/jpeg" -> "jpeg"
        extension = image_part.content_type.split("/")[-1]
        # Word 中的 JPEG 扩展名统一为 “jpg”
        if extension == "jpeg":
            extension = "jpg"

        # 全局图片计算器加1
        image_counter += 1

        # 生成图片文件名：原文件名_image_序号.扩展名
        image_name = f"{file_stem}_image_{image_counter}.{extension}"

        # 拼接完整输出路径
        image_path = image_output_dir / image_name

        # 将图片二进制数据写入文件
        image_path.write_bytes(image_part.blob)

        # 保存图片绝对路径
        image_paths.append(str(image_path.resolve()))

    return image_paths, image_counter


# ============================================================
# 第六步：定义自定义DOCX Loader
# ============================================================
class RichDocxLoader(BaseLoader):
    """
    将DOCX正文、表格行文本和图片引用按原始顺序组装成Document。
    每个DOCX文件产生一个Document。
    """

    def __init__(self, file_path: Path, image_output_dir: Path) -> None:
        """初始化加载器：保存DOCX路径和图片输出目录。"""
        self.file_path = file_path
        self.image_output_dir = image_output_dir

    def lazy_load(self) -> Iterator[Document]:
        """
        生成器：按DOCX原始块顺序读取文档，每个DOCX文件产生一个Document。
        使用yield逐个返回，符合BaseLoader接口。
        """

        # 创建图片输出目录（不存在则递归创建，已存在则忽略）
        self.image_output_dir.mkdir(parents=True, exist_ok=True)

        # 使用 python-docx 打开 Word 文档
        word_document = WordDocument(str(self.file_path))

        # 保存最终Document正文的各个部分（段落文本、表格描述、图片Markdown引用）
        content_parts: list[str] = []

        # 保存所有提取后的图片路径
        image_paths: list[str] = []

        # 识别到的表格数量
        table_count = 0

        # 图片计数器，用于生成唯一文件名
        image_counter = 0

        # 按照 DOCX 内部XML顺序遍历段落和表格（段落1 -> 表格1 -> 段落2 -> 表格2...）
        for block in iter_document_blocks(word_document):
            # ========================================================
            # 处理段落
            # ========================================================
            if isinstance(block, Paragraph):
                # 读取当前段落的普通文字
                paragraph_text = block.text.strip()
                # 有文字时，按当前位置加入正文
                if paragraph_text:
                    content_parts.append(paragraph_text)

                # 即使段落没有普通文字，也要检查其中是否包含图片
                # Word 中图片可能独占据一个段落，也可能和文字混合
                paragraph_images, image_counter = extract_paragraph_images(
                    paragraph=block,  # 当前段落
                    document=word_document,  # 当前文档
                    image_output_dir=self.image_output_dir,  # 图片输出目录
                    file_stem=self.file_path.stem,  # 原文件名（不扩展名）
                    image_counter=image_counter,  # 图片计数器
                )

                # 将当前段落中的图片按原位置加入正文
                for image_path in paragraph_images:
                    # 保存图片绝对路径
                    image_paths.append(image_path)
                    # 这样可以确保图片在Markdown中显示时，路径是相对的，而不是绝对的
                    # 例如：![嵌入图片1](image_1.jpg)
                    # 而不是：![嵌入图片1](/path/to/image_1.jpg)
                    image_number = len(image_paths)

                    # 生成Markdown格式的图片引用
                    content_parts.append(f"![嵌入图片{image_number}]({Path(image_path).name})")

            # ========================================================
            # 处理表格
            # ========================================================
            elif isinstance(block, Table):
                table_count += 1
                # 提取表格二维数据并转换为自然语言陈述句
                rows = extract_table_rows(block)
               # 输出的结果大概是这样的：
               # [
               #     ["商品", "单价", "库存"],  # 第 0 行（其实是表头，但 extract 不知道）
               #     ["耳机", "299", "10"],  # 第 1 行（其实是数据，extract 也不管）
               #     ["手机", "4999", "5"]  # 第 2 行
               # ]
                row_texts = table_rows_to_text(rows)

                if row_texts:
                    # 识别到表格时，按当前位置加入表格转换后的自然语言陈述句
                    content_parts.append(f"识别表格{table_count}：\n" + "\n".join(row_texts))

        # 生成一个Document
        yield Document(
            page_content="\n\n".join(content_parts),
            metadata={
                "source": str(self.file_path.resolve()), # 文档路径
                "table_count": table_count, # 识别到的表格数量
                "image_count": len(image_paths), # 识别到的图片数量
                "images": image_paths, # 图片绝对路径列表
            },
        )


# ============================================================
# 第七步：运行Loader并查看Document
# ============================================================
"""创建Loader，加载DOCX并打印Document结果。"""
loader = RichDocxLoader(
    file_path=FILE_PATH,
    image_output_dir=IMAGE_OUTPUT_DIR
)

# BaseLoader.load() 内部调用 lazy_load() 并收集所有Document
documents = loader.load()

print("Document数量：", len(documents))

for document in documents:
    print("识别表格数：", document.metadata["table_count"])
    print("识别图片数：", document.metadata["image_count"])
    print("图片文件：", [Path(path).name for path in document.metadata["images"]])
    print("\nDocument正文：")
    print(document.page_content)