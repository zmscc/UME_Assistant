from minio import Minio
from minio.error import S3Error
import logging
from fastapi import UploadFile
from typing import Dict, List
import os

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

    def save_files(self, files: List[UploadFile], knowledge_base_name: str, override: bool) -> List[Dict]:
        """
        保存多个文件到 MinIO。
        """
        results = []
        for file in files:
            result = self._save_single_file(file, knowledge_base_name, override)
            results.append(result)
        return results

    def _save_single_file(self, file: UploadFile, knowledge_base_name: str, override: bool) -> Dict:
        """
        内部方法：保存单个文件到 MinIO。
        """
        try:
            filename = file.filename
            bucket_name = knowledge_base_name
            object_name = filename

            data = {"knowledge_base_name": knowledge_base_name, "file_name": filename}

            # 检查文件是否已存在且大小相同
            file.seek(0, os.SEEK_END)
            content_length = file.tell()
            file.seek(0)

            if not override and self.get_object_size(bucket_name, object_name) == content_length:
                file_status = f"文件 {filename} 已存在。"
                self.logger.warning(file_status)
                return dict(code=409, msg=file_status, data=data)  # 修改为409表示冲突

            # 确保桶存在
            if not self.client.bucket_exists(bucket_name):
                self.client.make_bucket(bucket_name)

            # 上传文件到 MinIO
            self.client.put_object(
                bucket_name,
                object_name,
                file.file,  # 使用原始文件对象
                length=content_length,
                content_type=file.content_type
            )
            return dict(code=200, msg=f"成功上传文件 {filename}", data=data)
        except Exception as e:
            msg = f"{filename} 文件上传失败，报错信息为: {e}"
            self.logger.error(f"{e.__class__.__name__}: {msg}")
            return dict(code=500, msg=msg, data=data)


# 使用示例
if __name__ == "__main__":
    from fastapi import FastAPI, File, UploadFile
    from fastapi.responses import JSONResponse

    app = FastAPI()

    # 初始化 MinioStorage 类实例
    storage = MinioStorage(
        "192.168.150.101:9001",
        "kUeZWUIPvWvdz649",
        "Yfv1In0t75jYrkcOKj26GkYhcktxulHp",
        secure=False
    )

    @app.post("/upload/")
    async def upload_files(files: List[UploadFile] = File(...), knowledge_base_name: str = "default", override: bool = False):
        results = await storage.save_files(files, knowledge_base_name, override)
        return JSONResponse(content={"results": results})