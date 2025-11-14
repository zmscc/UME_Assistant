# ===================================================================================================
from pymilvus import connections, Collection, utility


def get_milvus_client(host="192.168.200.130", port="19530", collection_name="MyTest04"):
    # 建立到 Milvus 的连接
    connections.connect(
        alias="default",  # 连接别名，默认是 "default"
        host=host,
        port=port
    )

    # 获取集合对象
    if utility.has_collection(collection_name):
        collection = Collection(name=collection_name)
        return collection
    else:
        raise ValueError(f"Collection '{collection_name}' does not exist.")


# 创建 Milvus 客户端并获取集合对象
client = get_milvus_client()

# 示例：打印集合信息以确认连接成功
print("Connected to collection:", client.name)
# ===================================================================================================
