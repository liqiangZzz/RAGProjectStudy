from langchain_community.embeddings import DashScopeEmbeddings
from pymilvus import MilvusClient, DataType

from env_utils import DASHSCOPE_API_KEY

# ========== 1. 连接 Milvus ==========
MILVUS_URI = "http://127.0.0.1:19530"
COLLECTION_NAME = "rag_milvus_crud"

client = MilvusClient(
    uri=MILVUS_URI,
    collection_name=COLLECTION_NAME
    # user="root",        # 若开启了认证
    # password="Milvus",
)

# ========== 2. 初始化嵌入模型 ==========
embedding_model = DashScopeEmbeddings(
    model="text-embedding-v1",
    dashscope_api_key=DASHSCOPE_API_KEY,
)


def print_step(title: str):
    print("\n" + "=" * 25 + f" {title} " + "=" * 25)


def main() -> None:
    print("Milvus服务版本：", client.get_server_version())
    print("连接数据库：", COLLECTION_NAME)
    print("连接前集合：", client.list_collections())

    # 如果集合存在，先删除
    if client.has_collection(COLLECTION_NAME):
        client.drop_collection(COLLECTION_NAME)
        print(f"已删除旧集合：{COLLECTION_NAME}")

    try:
        # ====================================================
        # 第一步：建表与创建索引
        # ====================================================
        print_step("1. 建表与创建索引")
        # 确认向量维度
        dimension = len(embedding_model.embed_query("测试文本"))

        schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field("id", DataType.INT64, is_primary=True)
        schema.add_field("text", DataType.VARCHAR, max_length=1000)
        schema.add_field("source", DataType.VARCHAR, max_length=1000)
        # 向量字段
        schema.add_field("vector", DataType.FLOAT_VECTOR, dim=dimension)
        # 构建索引
        index_params = client.prepare_index_params()
        # 给向量字段添加索引
        index_params.add_index(
            field_name="vector",
            index_type="AUTOINDEX",
            metric_type="COSINE",  # 在向量检索时使用的相似度度量方式为“余弦相似度”
        )

        # 创建集合
        client.create_collection(
            collection_name=COLLECTION_NAME,
            schema=schema,
            index_params=index_params
        )

        print("建表完成：", COLLECTION_NAME, "| 向量维度：", dimension)

        # ====================================================
        # 第二步：插入数据
        # ====================================================
        print_step("2. 插入数据 (Insert)")
        rows = [
            {"id": 1, "text": "退款规则：仓库验收通过后发起原路退款，通常三个工作日内到账。", "source": "退款规则.md"},
            {"id": 2, "text": "物流规则：订单发货后可以在订单详情查看物流轨迹。", "source": "物流规则.md"},
            {"id": 3, "text": "退货规则：收货后七日内可以申请无理由退货。", "source": "退货规则.md"},
        ]

        # 将文本转换为向量表示
        vectors = embedding_model.embed_documents([row["text"] for row in rows])
        # 合并数据
        data = [{**row, "vector": vector} for row, vector in zip(rows, vectors)]
        # 插入数据
        client.insert(data=data, collection_name=COLLECTION_NAME)
        # 确保插入数据已经写入 Milvus 数据库
        client.flush(COLLECTION_NAME)
        # 加载Collection
        client.load_collection(COLLECTION_NAME)

        print("插入实体数：", client.get_collection_stats(COLLECTION_NAME).get("row_count"))

        # 第四步：向量检索
        # ====================================================
        query_vector = embedding_model.embed_query("退款什么时候到账？")
        hits = client.search(
            collection_name=COLLECTION_NAME,
            data=[query_vector],
            anns_field="vector",  # 显式指定向量字段。好处：防止未来表结构升级（如多模态场景有多个向量字段）时产生歧义。
            search_params={"metric_type": "COSINE"},  # 显式指定距离度量。好处：必须与建索引时的 metric_type 严格一致，同时为未来调整检索性能参数（如 ef, nprobe）留出扩展空间。
            limit=3,
            output_fields=["id", "text", "source"],
        )

        # data:[
        #     [
        #         {'id': 1, 'distance': 0.7756346464157104,
        #          'entity': {'text': '退款规则：仓库验收通过后发起原路退款，通常三个工作日内到账。',
        #                     'source': '退款规则.md', 'id': 1}},
        #         {'id': 3, 'distance': 0.6084451675415039,
        #          'entity': {'text': '退货规则：收货后七日内可以申请无理由退货。', 'source': '退货规则.md', 'id': 3}},
        #         {'id': 2, 'distance': 0.33424896001815796,
        #          'entity': {'text': '物流规则：订单发货后可以在订单详情查看物流轨迹。', 'source': '物流规则.md', 'id': 2}}
        #     ]
        # ]
        # print("向量检索结果：",hits)

        for hit in hits[0]:
            print("hit:", hit)
            entity = hit["entity"]
            print("- id=", entity["id"], "score=", round(hit["distance"], 4), "|", entity["text"])

        # ====================================================
        # 第四步：标量过滤 (Query)
        # ====================================================
        print_step("4. 标量过滤 (Scalar Query)")
        resp = client.query(
            collection_name=COLLECTION_NAME,
            filter='source == "退款规则.md"',
            output_fields=["id", "text", "source"],
        )
        print("标量过滤结果：")
        for item in resp:
            print("- id=", item["id"], "source=", item["source"], "|", item["text"])

        # ====================================================
        # 第五步：更新数据 (Upsert)
        # ====================================================
        print_step("5. 更新数据 (Upsert)")
        client.upsert(
            collection_name=COLLECTION_NAME,
            data=[
                {
                    "id": 2,
                    "text": "物流规则：订单发货后可以在订单详情查看物流轨迹，签收后自动完成。",
                    "source": "物流规则.md",
                    "vector": embedding_model.embed_query(
                        "物流规则：订单发货后可以在订单详情查看物流轨迹，签收后自动完成。"
                    ),
                }
            ]
        )
        # 确保更新的数据已经写入Milvus
        client.flush(COLLECTION_NAME)
        # 获取更新后的实体数
        updated_data = client.get(
            collection_name=COLLECTION_NAME,
            ids=[2],
            output_fields=["id", "text", "source", ],
        )
        print("更新结果（id=2）：")
        for item in updated_data:
            print("- id=", item["id"], "source=", item["source"], "|", item["text"])

        # ====================================================
        # 第六步：删除数据 (Delete)
        # ====================================================
        print_step("6. 删除数据 (Delete)")
        client.delete(collection_name=COLLECTION_NAME, ids=[1])
        # 确保删除的数据已经写入Milvus
        client.flush(COLLECTION_NAME)
        remaining_data = client.query(
            collection_name=COLLECTION_NAME,
            filter="id > 0",
            output_fields=["id", "source"],
        )
        print("删除后剩余：")
        for item in remaining_data:
            print("- id=", item["id"], "source=", item["source"])
    finally:
        client.close()
        print("\n[系统提示] Milvus 连接已安全关闭。")


if __name__ == '__main__':
    main()
