from pymilvus import MilvusClient, DataType, Function, FunctionType

MILVUS_URI = "http://localhost:19530"
MILVUS_DB_NAME = "default"
COLLECTION_NAME = "tmp_course_bm25"


def create_collection(client: MilvusClient):
    """创建包含文本字段和BM25稀疏向量的集合"""

    # 如果Collection已存在，先删除
    if client.has_collection(collection_name=COLLECTION_NAME):
        client.drop_collection(collection_name=COLLECTION_NAME)

    # 创建自定义 Schema
    schema = client.create_schema(
        # 主键（id 字段）由 Milvus 自动生成（递增整数）
        auto=True,
        # 不允许动态添加字段，插入数据时只能使用 schema 中定义的字段，不能额外添加未定义的字段。
        enable_dynamic_field=False
    )

    # 主键字段有 Milvus 自动生成，递增整数
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True, auto_id=True)

    # BM25 的输入的文本字段
    schema.add_field(
        field_name="text",
        datatype=DataType.VARCHAR,
        max_length=1024,

        # 当这个字段的数据要转成向量（或者倒排索引）时，请用 jieba 把它切碎
        enable_analyzer=True,  # 开启文本分析功能，BM25需要使用
        analyzer_params={"tokenizer": "jieba"},  # 中文使用 jieba 分词
    )

    # 文档来源
    schema.add_field(
        field_name="source",
        datatype=DataType.VARCHAR,
        max_length=200,
    )

    # 保存 BM25 产生的稀疏向量（高维向量，每个维度的值为 0 或 1，0 表示该维度不存在，1 表示该维度存在）,大多数位置为 0
    # 该字段无需制定维度，稀疏向量由输入的文本字段自动生成
    schema.add_field(
        field_name="sparse_vector",
        datatype=DataType.SPARSE_FLOAT_VECTOR,
    )

    # 定义BM25函数： text字段自动转换成 sparse_vector 字段
    # 函数作用： 当向集合插入数据时，只需提供 text，Milvus 会自动根据  BM25 算法（基于集合内所有文档的词频统计）
    # 计算稀疏向量并存入 sparse_vector 字段。查询时也可以用原始文本作为查询向量，由引擎内部转换为稀疏向量进行检索。
    schema.add_function(
        Function(
            name="b25_vector",  # 自定义函数名称
            function_type=FunctionType.BM25,  # 指定为 BM25 转换函数
            input_field_names=["text"],  # 输入字段为 text
            output_field_names=["sparse_vector"],  # 输出字段为 sparse_vector
        )
    )

    # 为BM25稀疏向量创建倒排索引
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="sparse_vector",  # 索引字段为 sparse_vector
        index_type="SPARSE_INVERTED_INDEX",  ## 稀疏向量专用倒排索引，适合 BM25 相似度计算
        metric_type="BM25",  # 指定为 BM25 相似度计算相关性分数
    )

    # 创建Collection
    client.create_collection(
        collection_name=COLLECTION_NAME,  # 指定Collection名称
        schema=schema,  # 指定Collection的Schema
        index_params=index_params,  # 指定Collection的索引参数
    )


client = MilvusClient(
    url=MILVUS_URI,  # Milvus 服务地址
    db_name=MILVUS_DB_NAME,  # Milvus 数据库名称
)

# 创建Collection
create_collection(client)

# 只插入原始文本，不需要提供sparse_vector字段
client.insert(
    collection_name=COLLECTION_NAME,
    data=[
        {"text": "AB-120 是降噪耳机型号。", "source": "产品型号.md", },
        {"text": "降噪耳机退货后，退款通常三个工作日内到账。", "source": "退款规则.md", },
        {"text": "AB-121 是运动耳机型号。", "source": "产品型号.md", },
        {"text": "订单发货后可以在订单详情查看物流轨迹。", "source": "物流规则.md", },
    ]
)

# 强制刷新内存中的数据到磁盘，确保数据已持久化，后续操作能立即看到最新数据
client.flush(COLLECTION_NAME)

# 将集合加载到内存中，只有加载后才能进行查询操作
client.load_collection(COLLECTION_NAME)

# ===========稀疏向量检索===============
question = "AB-120 退货后多久到账？"
results = client.search(
    collection_name=COLLECTION_NAME,
    data=[question],
    anns_field="sparse_vector",
    search_params={"metric_type": "BM25"},
    limit=3,
    output_fields=["text", "source"]
)

print("稀疏向量检索结果:", results)
print("-" * 80)

# ===========普通查询===============
results = client.query(
    collection_name=COLLECTION_NAME,
    filter='text like "%退货%"',
    output_fields=["text", "source"]
)
print("普通查询结果:", results)
