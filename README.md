# RAGProjectStudy

RAG（Retrieval-Augmented Generation，检索增强生成）学习项目，基于 LangChain 与 LangGraph，按学习阶段分为五个章节，每章为独立 Python 包，内含按编号组织的示例脚本。模型接入：DeepSeek / GLM（LangChain `init_chat_model`）、DashScope（重排序与 Embedding）。

## 目录结构

```
RAGProjectStudy/
├── 01_RAG介绍与三种架构/        # RAG 基础概念与架构模式
│   ├── 01_two_step_rag.py       # 两步式（朴素）RAG
│   ├── 02_agentic_tag.py        # Agentic RAG
│   ├── 03_hybrid_rag.py         # 混合检索 RAG
│   └── images/                  # 配图
├── 02_LangChain RAG构建流程/    # LangChain RAG 核心组件与构建流程
│   ├── 01_load_txt.py           # 文本加载
│   ├── 02_custom_txt_loader.py  # 自定义 Loader
│   ├── 03_load_rich_pdf.py      # PDF 加载
│   ├── 04_load_rich_docx.py     # Word 加载
│   ├── 05_embedding_semantic_splitter.*  # 语义切分（含笔记 md 与示意图）
│   ├── 06_inmemory_vectorstore.py   # 内存向量库
│   ├── 07_milvus_vectorstore.py     # Milvus 向量库
│   ├── 08_inmemory_retriever_modes.py   # Retriever 模式（内存）
│   ├── 09_milvus_retriever_modes.py     # Retriever 模式（Milvus）
│   ├── 10_knowledge_base_comprehensive.py   # 知识库综合示例
│   └── data/                    # 示例数据
├── 03_LangChainRAG应用/         # LangChain RAG 应用实践
│   ├── 01_langchain_retrieval.py            # 检索基础
│   ├── 02_milvus_bm25_search.py             # BM25 稀疏检索
│   ├── 03_milvus_hybrid_search.py           # 混合检索
│   ├── 04_milvus_hybrid_search_rag.py       # 混合检索 RAG
│   ├── 05_dashscope_rerank.py               # DashScope 重排序
│   ├── 06_milvus_hybrid_dashscope_rerank.py # 混合检索 + 重排序
│   ├── 07_query_transformations.py          # 查询改写
│   ├── 08_create_agent_rag1.py              # Agent RAG（一）
│   └── 09_create_agent_rag2.py              # Agent RAG（二）
├── 04_LangGraph RAG使用/        # LangGraph 高级 RAG
│   ├── 01_fixed_retrieval_self_correcting_rag.py  # 固定检索自校正 RAG
│   ├── 02_adaptive_retrieval_self_correction_rag.py # 自适应检索自校正 RAG
│   ├── 03_langgraph_hybrid_rag.py               # LangGraph 混合 RAG
│   └── images/
├── 05_RAG评估/                  # RAG 检索评估指标（模拟 LLM Judge，不接真实 API）
│   └── 01_评估检索层/           # 检索层（Retrieval）评估
│       ├── 01_hit_at_k_case.py               # Hit@K（命中率）
│       ├── 02_precision_at_k_case.py         # Precision@K（精确率）
│       ├── 03_recall_at_k_case.py            # Recall@K（召回率）
│       ├── 04_mrr_case.py                    # MRR（平均倒数排名）
│       ├── 05_ndcg_at_k_case.py              # NDCG@K（归一化折损累计增益）
│       ├── 06_context_precision_at_k_case.py # Context Precision@K（上下文精确率，RAGAS 口径）
│       └── 07_context_recall_at_k_case.py    # Context Recall@K（上下文召回率，RAGAS 口径）
├── env_utils.py                 # 环境变量读取工具
├── init_llm.py                  # LLM 初始化
├── requirements.txt             # 依赖清单（含全部锁定版本，一键安装）
└── .env.example                 # 环境变量模板
```

## 编号规范

- 所有示例脚本按 `两位编号_主题.py` 命名（如 `01_load_txt.py`），编号即推荐学习顺序。
- 每个章节目录内含 `__init__.py`，作为独立 Python 包。
- 图文笔记随章节存放：`*.md` 为文字笔记，`images/` 与 `*.png` 为示意图。

## 环境准备

```bash
# 克隆项目
git clone git@github.com:liqiangZzz/RAGProjectStudy.git
cd RAGProjectStudy

# 创建虚拟环境（建议 Python 3.12+，生成环境为 3.14.x）
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip

# 一键安装全部依赖（requirements.txt 已锁定全部 101 个包的版本）
pip install -r requirements.txt

# 配置环境变量
cp .env.example .env
# 编辑 .env 填入 API Key：DEEPSEEK_API_KEY / GLM_API_KEY / DASHSCOPE_API_KEY 等
```

> 注意：
> - `.env` 包含密钥，已被 `.gitignore` 过滤，不会提交到远程仓库，新电脑需用 `.env.example` 模板重新配置。
> - `milvus-lite` / `faiss-cpu` 无 Windows 原生包，Windows 用户建议使用 WSL2，或让 pymilvus 直连远程 Milvus 服务器。

## 评估指标速览（05_RAG评估）

| 指标 | 中文名 | 回答的问题 | 排名敏感 |
|---|---|---|---|
| Hit@K | 命中率 | 前 K 条里有没有命中相关来源 | 否 |
| Precision@K | 精确率 | 前 K 条里命中了几个（结果干净度） | 否 |
| Recall@K | 召回率 | 该找的相关来源找全了没有 | 否 |
| MRR | 平均倒数排名 | 第一个相关来源排得多靠前 | 是 |
| NDCG@K | 归一化折损累计增益 | 整体排序质量（log2 位置折损 + 理想排序归一化） | 是 |
| Context Precision@K | 上下文精确率（RAGAS） | 有用的切片是否排在靠前位置（Chunk 级 verdict） | 是 |
| Context Recall@K | 上下文召回率（RAGAS） | 参考答案的事实声明被前 K 条切片支撑的比例（Claim 级归因） | 否 |

## 运行示例

```bash
python "02_LangChain RAG构建流程/01_load_txt.py"
```
