'''这个脚本提供了一个用于表示和管理对话历史记录的工具'''
import logging
from functools import lru_cache
from typing import Dict, List, Tuple, Union

from langchain.prompts.chat import ChatMessagePromptTemplate

from chatchat.server.pydantic_v2 import BaseModel, Field
from chatchat.utils import build_logger


logger = build_logger()


class History(BaseModel):
    """
    对话历史
    可从dict生成，如
    h = History(**{"role":"user","content":"你好"})
    也可转换为tuple，如
    h.to_msy_tuple = ("human", "你好")
    """

    role: str = Field(...) # Field(...) 是 pydantic 库的一个功能，用于为字段提供额外的配置，比如默认值、别名、验证器等。
    content: str = Field(...)

    def to_msg_tuple(self):
        return "ai" if self.role == "assistant" else "human", self.content

    def to_msg_template(self, is_raw=True) -> ChatMessagePromptTemplate:
        role_maps = {
            "ai": "assistant",
            "human": "user",
        }
        role = role_maps.get(self.role, self.role)
        if is_raw:  # 当前默认历史消息都是没有input_variable的文本。
            content = "{% raw %}" + self.content + "{% endraw %}"
        else:
            content = self.content

        return ChatMessagePromptTemplate.from_template(
            content,
            "jinja2",
            role=role,
        )

    @classmethod # 类方法。第一个参数cls表示类本身；此方法可以通过类直接调用，而不需要实例化对象。
    def from_data(cls, h: Union[List, Tuple, Dict]) -> "History": # h：可以是 List、Tuple 或 Dict 类型的数据，表示要转换成 History 实例的原始数据。
        if isinstance(h, (list, tuple)) and len(h) >= 2: # 如果 h 是列表或元组，并且长度至少为 2，则认为前两个元素分别对应 role 和 content 属性。
            h = cls(role=h[0], content=h[1])
        elif isinstance(h, dict): # h是字典，使用字典解包 (**h) 将键值对作为关键字参数传递给 History 构造函数，创建一个新的 History 实例，并将其赋值给 h。
            '''
            # 假设有一个字典
            h = {"role": "user", "content": "Hello, how are you?"}
            # 使用字典解包创建 History 实例
            h = History(**h)
            # 等价于
            h = History(role="user", content="Hello, how are you?")
            '''
            h = cls(**h)

        return h
