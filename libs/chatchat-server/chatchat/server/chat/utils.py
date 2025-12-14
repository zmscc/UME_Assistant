'''这个脚本提供了一个用于表示和管理对话历史记录的工具'''
import logging
from functools import lru_cache
from typing import Dict, List, Tuple, Union

from langchain.prompts.chat import ChatMessagePromptTemplate

from chatchat.server.pydantic_v2 import BaseModel, Field
from chatchat.utils import build_logger


logger = build_logger()


class History(BaseModel):
    role: str = Field(...) # Field(...)在 Pydantic 中表示该字段是必需的，不能缺失。
    content: str = Field(...)

    def to_msg_tuple(self): # 创建一个二元组，用于表示对话历史消息。
        return "ai" if self.role == "assistant" else "human", self.content

    def to_msg_template(self, is_raw=True) -> ChatMessagePromptTemplate: # 把一条对话历史消息转换成一个可用于提示词（prompt）模板的对象
        role_maps = {
            "ai": "assistant",
            "human": "user",
        }
        role = role_maps.get(self.role, self.role) # dict.get(键, 默认值)，如果键存在，就返回对应的值；如果键不存在，则返回默认值。
        if is_raw:
            content = "{% raw %}" + self.content + "{% endraw %}" # {% raw %}...{% endraw %} 包起来，给到jinja2处理时表示这段内容请原样输出，不要当模板解析！
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
