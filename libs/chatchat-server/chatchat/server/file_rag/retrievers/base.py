from __future__ import annotations # 这行导入使得类型注解中的前向引用（forward references）可以被当作字符串处理，避免在类内部定义时出现名称解析问题。


from abc import ABCMeta, abstractmethod # 导入用于定义抽象基类和抽象方法的工具。

from langchain.vectorstores import VectorStore # 导入用于存储和检索向量数据的类

'''
BaseRetrieverService 是一个设计模式的一部分，它通过定义一组标准接口来规范所有具体检索服务的行为。任何继承自 BaseRetrieverService 的子类都必须实现这些抽象方法，
确保它们遵循相同的规则来进行初始化、从向量存储中构建检索器以及基于查询检索相关文档。这种设计提高了代码的可维护性和扩展性，因为新增加的检索服务只需要遵循已定义的接口即可。
'''
class BaseRetrieverService(metaclass=ABCMeta):
    def __init__(self, **kwargs):
        self.do_init(**kwargs)

    @abstractmethod # 这是一个抽象方法，子类必须实现它以完成具体的初始化逻辑。
    def do_init(self, **kwargs):
        pass

    @abstractmethod # 抽象方法，定义了如何从给定的 VectorStore 实例创建检索器。它接受三个参数：
    def from_vectorstore(
        vectorstore: VectorStore, # 存储向量数据的对象。
        top_k: int, # 返回最相关的文档数量。
        score_threshold: int | float, # 相似度评分的阈值，只有分数高于此值的文档才会被认为是相关的。
    ):
        pass

    @abstractmethod # 抽象方法，定义了根据查询字符串获取相关文档的方法。具体实现将负责执行实际的检索逻辑。
    def get_relevant_documents(self, query: str):
        pass
