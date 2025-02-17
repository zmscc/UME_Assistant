from __future__ import annotations

import json
import re
from typing import Any, Callable, Dict, Optional, Tuple, Type, Union

from langchain.agents import tool
from langchain_core.tools import BaseTool

from chatchat.server.knowledge_base.kb_doc_api import DocumentWithVSId
from chatchat.server.pydantic_v1 import BaseModel, Extra

'''这份代码定义了一个工具注册和管理的框架，定义了一个工具注册系统，允许开发者将函数作为工具注册到系统中，并可以对这些工具进行管理。'''

__all__ = ["regist_tool", "BaseToolOutput", "format_context"]


_TOOLS_REGISTRY = {} # 用于存储所有注册的工具的字典


# patch BaseTool to support extra fields e.g. a title
BaseTool.Config.extra = Extra.allow # 修改 BaseTool 配置，允许 BaseTool 模型接受额外的字段。

################################### TODO: workaround to langchain #15855
# patch BaseTool to support tool parameters defined using pydantic Field


def _new_parse_input(
    self,
    tool_input: Union[str, Dict],
) -> Union[str, Dict[str, Any]]:
    """Convert tool input to pydantic model."""
    input_args = self.args_schema
    if isinstance(tool_input, str):
        if input_args is not None:
            key_ = next(iter(input_args.__fields__.keys()))
            input_args.validate({key_: tool_input})
        return tool_input
    else:
        if input_args is not None:
            result = input_args.parse_obj(tool_input)
            return result.dict()


def _new_to_args_and_kwargs(self, tool_input: Union[str, Dict]) -> Tuple[Tuple, Dict]:
    # For backwards compatibility, if run_input is a string,
    # pass as a positional argument.
    if isinstance(tool_input, str):
        return (tool_input,), {}
    else:
        # for tool defined with `*args` parameters
        # the args_schema has a field named `args`
        # it should be expanded to actual *args
        # e.g.: test_tools
        #       .test_named_tool_decorator_return_direct
        #       .search_api
        if "args" in tool_input:
            args = tool_input["args"]
            if args is None:
                tool_input.pop("args")
                return (), tool_input
            elif isinstance(args, tuple):
                tool_input.pop("args")
                return args, tool_input
        return (), tool_input


# 替换 BaseTool 的 _parse_input 和 _to_args_and_kwargs 方法，增加了对工具参数的解析和处理功能。
BaseTool._parse_input = _new_parse_input
BaseTool._to_args_and_kwargs = _new_to_args_and_kwargs
###############################

'''装饰器的定义：装饰器是一个接受一个函数作为参数并返回一个新函数的函数。注解只是应用装饰器的一种便捷方式。'''
'''这个装饰器函数用于将一个函数注册为一个工具，并添加到 _TOOLS_REGISTRY 中。'''
def regist_tool(
    *args: Any, # 接受任意数量的位置参数
    title: str = "", # 工具的标题，默认为空字符串
    description: str = "",
    return_direct: bool = False, # 指示工具的返回值是否直接可用，默认为 False。
    args_schema: Optional[Type[BaseModel]] = None, # 可选的参数模式，用于验证工具的输入，默认为 None。
    infer_schema: bool = True, # 是否自动推断参数模式，默认为 True。
) -> Union[Callable, BaseTool]: # 返回值可以是 Callable（装饰器）或 BaseTool（工具实例）。
    """
    wrapper of langchain tool decorator
    add tool to regstiry automatically
    """
    # 解析和配置一个BaseTool对象
    def _parse_tool(t: BaseTool):
        nonlocal description, title # 使用 nonlocal 关键字来修改外部函数 regist_tool 中的 description 和 title 变量。

        _TOOLS_REGISTRY[t.name] = t

        # change default description
        if not description:
            if t.func is not None:
                description = t.func.__doc__
            elif t.coroutine is not None:
                description = t.coroutine.__doc__
        t.description = " ".join(re.split(r"\n+\s*", description))
        # set a default title for human
        if not title:
            title = "".join([x.capitalize() for x in t.name.split("_")])
        t.title = title

    def wrapper(def_func: Callable) -> BaseTool: # 对wrapper函数的理解见text2sql.py文件最后的注解
        partial_ = tool(
            *args,
            return_direct=return_direct,
            args_schema=args_schema,
            infer_schema=infer_schema,
        )
        t = partial_(def_func) # 调用 partial_(def_func) 来创建一个 BaseTool 对象 t。
        _parse_tool(t) # 调用 _parse_tool(t) 来配置工具的描述和标题。
        return t # 返回配置好的 BaseTool 对象 t

    if len(args) == 0: # 如果 args 为空，表示 regist_tool 被用作装饰器，因此返回 wrapper 函数本身。
        return wrapper
    else: # 如果 args 不为空，表示 regist_tool 被直接调用并传递了参数，使用 tool 装饰器创建一个 BaseTool 对象 t，然后调用 _parse_tool(t) 来配置它，并返回配置好的 BaseTool 对象。
        t = tool( # 创建一个 BaseTool 实例
            *args,
            return_direct=return_direct,
            args_schema=args_schema,
            infer_schema=infer_schema,
        )
        _parse_tool(t) # 使用 _parse_tool 来配置它
        return t # 返回BaseTool

'''这个类用于封装工具的输出，以便于在不同的上下文中使用。
它可以将数据转换为字符串或 JSON 格式，也可以通过自定义的格式化函数来转换。'''
class BaseToolOutput:
    """
    LLM 要求 Tool 的输出为 str，但 Tool 用在别处时希望它正常返回结构化数据。
    只需要将 Tool 返回值用该类封装，能同时满足两者的需要。
    基类简单的将返回值字符串化，或指定 format="json" 将其转为 json。
    用户也可以继承该类定义自己的转换方法。
    """

    def __init__( # 构造函数
        self,
        data: Any, # 任何类型的数据，这是工具返回的实际数据。
        format: str | Callable = None, # 指定如何将 data 转换为字符串。如果设置为 "json"，则将 data 转换为 JSON 字符串。如果是一个可调用对象，则使用该对象进行转换。
        data_alias: str = "",
        **extras: Any, # 关键字参数，用于存储额外的数据。
    ) -> None:
        self.data = data
        self.format = format
        self.extras = extras
        if data_alias: # 如果提供了 data_alias，则创建一个属性，其值为 data
            setattr(self, data_alias, property(lambda obj: obj.data))

    def __str__(self) -> str: # 定义了如何将 BaseToolOutput 实例转换为字符串
        if self.format == "json": # 如果 format 是 "json"，则使用 json.dumps 将 data 转换为 JSON 字符串。
            return json.dumps(self.data, ensure_ascii=False, indent=2)
        elif callable(self.format): # 如果 format 是一个可调用对象，则调用该对象，并将 self 作为参数传递。
            return self.format(self)
        else: # 如果 format 不是上述两种情况，则简单地将 data 转换为字符串。
            return str(self.data)


'''
format_context 的方法是 BaseToolOutput 类的一个成员函数，主要作用是将包含知识库输出的 ToolOutput 格式化为 LLM（大型语言模型）需要的字符串格式
eg:
[
    {"id": 1, "page_content": "这是第一个文档的内容"},
    {"id": 2, "page_content": "这是第二个文档的内容"}
]
调用format_context后:
这是第一个文档的内容


这是第二个文档的内容
'''
def format_context(self: BaseToolOutput) -> str:
    '''
    将包含知识库输出的ToolOutput格式化为 LLM 需要的字符串
    '''
    context = "" # 初始化为空字符串，用于存储最终格式化后的文本。
    docs = self.data["docs"] # 从 self.data 中提取出 docs 键对应的值
    source_documents = [] # 存储提取出来的文档内容。

    for inum, doc in enumerate(docs):
        doc = DocumentWithVSId.parse_obj(doc)
        source_documents.append(doc.page_content)

    if len(source_documents) == 0:
        context = "没有找到相关文档,请更换关键词重试"
    else:
        for doc in source_documents:
            context += doc + "\n\n"

    return context
