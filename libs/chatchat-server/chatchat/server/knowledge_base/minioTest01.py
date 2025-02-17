from minio import Minio
from minio.error import S3Error
import os
import logging
from typing import Dict

class MinioStorage:
    def __init__(self, endpoint: str, access_key: str, secret_key: str, secure: bool = True):
        """初始化 MinIO 存储客户端"""
        self.client = Minio(endpoint, access_key=access_key, secret_key=secret_key, secure=secure)
        self.logger = logging.getLogger(__name__)

    def get_object_size(self, bucket_name: str, object_name: str) -> int:
        """获取 MinIO 中对象的大小"""
        try:
            obj = self.client.stat_object(bucket_name, object_name)
            return obj.size
        except S3Error as err:
            if err.code == 'NoSuchKey':
                return 0
            raise

    def save_file(self, file, filename: str, knowledge_base_name: str, override: bool, content_length: int) -> Dict:
        """
        保存单个文件到 MinIO。
        """
        try:
            bucket_name = knowledge_base_name
            object_name = filename

            data = {"knowledge_base_name": knowledge_base_name, "file_name": filename}

            # 检查文件是否已存在且大小相同
            if not override and self.get_object_size(bucket_name, object_name) == content_length:
                file_status = f"文件 {filename} 已存在。"
                self.logger.warning(file_status)
                return dict(code=409, msg=file_status, data=data)  # 修改为409表示冲突

            # 确保桶存在
            if not self.client.bucket_exists(bucket_name):
                self.client.make_bucket(bucket_name)

            # 重置文件指针到文件开头
            file.seek(0)

            # 上传文件到 MinIO
            self.client.put_object(
                bucket_name,
                object_name,
                file,
                length=content_length,
                content_type="application/octet-stream"  # 如果需要特定的内容类型，可以在调用时指定
            )
            return dict(code=200, msg=f"成功上传文件 {filename}", data=data)
        except Exception as e:
            msg = f"{filename} 文件上传失败，报错信息为: {e}"
            self.logger.error(f"{e.__class__.__name__}: {msg}")
            return dict(code=500, msg=msg, data=data)

    def upload_file_from_path(self, file_path: str, knowledge_base_name: str, override: bool = False) -> Dict:
        """
        从给定路径加载文件并上传到 MinIO。
        """
        try:
            # 获取文件大小而不将文件内容全部读入内存
            content_length = os.path.getsize(file_path)

            with open(file_path, 'rb') as file:
                filename = os.path.basename(file_path)
                return self.save_file(file, filename, knowledge_base_name, override, content_length)
        except Exception as e:
            self.logger.error(f"文件上传过程中出现错误: {e}")
            return dict(code=500, msg=f"文件上传过程中出现错误: {e}", data={})

# 主函数示例：使用 MinioStorage 类上传文件
if __name__ == "__main__":
    # 初始化 MinioStorage 类实例
    storage = MinioStorage(
        "192.168.150.101:9000",
        "kUeZWUIPvWvdz649",
        "Yfv1In0t75jYrkcOKj26GkYhcktxulHp",
        secure=False
    )

    # 定义要上传的文件路径、知识库名称和是否覆盖选项
    file_path = "C:/zuoye.PNG"  # 替换为你的文件路径
    knowledge_base_name = "ume"  # 替换为你的知识库名称
    override = False  # 设置是否覆盖已有文件

    # 调用上传文件的方法
    result = storage.upload_file_from_path(file_path, knowledge_base_name, override)
    print(result)