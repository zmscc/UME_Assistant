from langchain.vectorstores import Milvus
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, utility

# 定义连接参数
# alias = "default"  # 连接别名
# host = "192.168.150.101"  # Milvus服务IP地址
# port = "19530"  # Milvus服务端口
# user = "root"  # 用户名
# password = "milvus@UME"  # 密码

host = "39.105.147.191"  # Milvus服务IP地址
port = "19530"  # Milvus服务端口
user = ""  # 用户名
password = ""  # 密码

# 尝试连接到Milvus
try:
    connections.connect(
        host=host,
        port=port,
        user=user,  # 如果您的pymilvus版本支持此参数
        password=password  # 如果您的pymilvus版本支持此参数
    )
    print("成功连接到Milvus")

except Exception as e:
    print(f"连接失败: {e}")