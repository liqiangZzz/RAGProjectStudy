from pathlib import Path
from typing import Iterator

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document


class TextFileLoader(BaseLoader):

    def __init__(self, file_path: Path, encoding: str = "utf-8"):
        self.file_path = file_path
        self.encoding = encoding

    def lazy_load(self) -> Iterator[Document]:
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")

        content = self.file_path.read_text(encoding=self.encoding)

        yield Document(
            page_content=content,
            metadata={
                "source": str(self.file_path),
                "file_name": str(self.file_path.name),
                "file_type": self.file_path.suffix,
                "encoding": self.encoding,
            }
        )


FILE_PATH = Path(
    r"/Users/Python/project/project-learn/python-code/RAGProjectStudy/02_LangChain RAG构建流程/data/售后规则.txt")

loader = TextFileLoader(file_path=FILE_PATH, encoding="utf-8")

documents = loader.load()

for index, document in enumerate(documents):
    print(f"document {index}: {document}")
