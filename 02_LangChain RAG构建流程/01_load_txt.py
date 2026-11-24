from langchain_community.document_loaders import TextLoader

loader = TextLoader(
    file_path=r"/Users/Python/project/project-learn/python-code/RAGProjectStudy/02_LangChain RAG构建流程/data/售后规则.txt",
    encoding="utf-8",
)

documents = loader.load()

print("documents:", documents)

for index,document in enumerate(documents,start=1):
    print(f"document {index}: {document}")