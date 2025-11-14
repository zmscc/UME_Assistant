'''这个脚本提供了一个接口，用户可以通过它发送聊天请求，然后脚本根据请求中的参数决定是直接与 LLM 模型对话，还是通过代理与多个工具进行交互，或者直接调用单个工具'''
from __future__ import annotations

from typing import Dict, List

from fastapi import APIRouter, Request
from langchain.prompts.prompt import PromptTemplate
from sse_starlette import EventSourceResponse

from chatchat.server.api_server.api_schemas import OpenAIChatInput
from chatchat.server.chat.chat import chat
from chatchat.server.chat.kb_chat import kb_chat
from chatchat.server.chat.feedback import chat_feedback
from chatchat.server.chat.file_chat import file_chat
from chatchat.server.db.repository import add_message_to_db
from chatchat.server.utils import (
    get_OpenAIClient,
    get_prompt_template,
    get_tool,
    get_tool_config,
)
from chatchat.settings import Settings
from chatchat.utils import build_logger
from .openai_routes import openai_request, OpenAIChatOutput


logger = build_logger()

'''
chat_router 是一个专门处理与聊天对话相关的 API 端点的路由器。它本身并不直接处理请求，而是作为一个容器来组织相关路由。
可以在这个路由器上定义多个路径操作函数（如 GET、POST 等），这些函数将被挂载到 /chat 前缀下。
'''
# APIRouter()：这是一个路由器对象，允许你将一组相关的路由逻辑组织在一起。你可以为不同的功能模块创建多个路由器，然后将它们包含到主应用中（FastAPI）。
chat_router = APIRouter(prefix="/chat", tags=["ChatChat 对话"]) # prefix="/chat"：为该路由器下的所有路由添加一个公共前缀 /chat。这意味着所有通过这个路由器定义的端点都会以 /chat 开头。

# chat_router.post(
#     "/chat",
#     summary="与llm模型对话(通过LLMChain)",
# )(chat)

chat_router.post(
    "/feedback",
    summary="返回llm模型对话评分",
)(chat_feedback) # 将 chat_feedback 函数注册为处理 /feedback 路径的 HTTP POST 请求的处理器。chat_feedback的具体实现决定了如何处理接收到的反馈数据，并返回相应的响应。


chat_router.post("/kb_chat", summary="知识库对话")(kb_chat)
chat_router.post("/file_chat", summary="文件对话")(file_chat)



'''Agent入口(上面定义路由的统一聊天接口，支持多种类型的对话处理)'''
'''此装饰函数等价于chat_router.post("/chat/completions", summary="兼容 openai 的统一 chat 接口")(chat_completions)'''
@chat_router.post("/chat/completions", summary="兼容 openai 的统一 chat 接口") # @chat_router.post(...)：这是一个装饰器，用来将 chat_completions 函数注册为处理 /chat/completions 路径的 HTTP POST 请求的处理器。
async def chat_completions(
    request: Request, # 这个参数非必要，FastAPI 已经将请求体中的数据通过 Pydantic 模型 OpenAIChatInput 解析并传递给了 body 参数，所以通常情况下不需要直接访问原始请求对象
    body: OpenAIChatInput, # 定义了 body: OpenAIChatInput 作为参数，客户端发送的 POST 请求必须包含符合 OpenAIChatInput 模型结构的 JSON 数据。FastAPI 会自动解析并验证这个 JSON 数据，确保其符合模型定义。如果不符合，FastAPI 会返回一个详细的错误响应给客户端。
) -> Dict:
    """
    请求参数与 openai.chat.completions.create 一致，可以通过 extra_body 传入额外参数
    tools 和 tool_choice 可以直接传工具名称，会根据项目里包含的 tools 进行转换
    通过不同的参数组合调用不同的 chat 功能：
    - tool_choice:工具对话
        - extra_body 中包含 tool_input: 直接调用 tool_choice(tool_input)
        - extra_body 中不包含 tool_input: 通过 agent 调用 tool_choice
    - tools: agent 对话
    - 其它：LLM 对话(没有指定 tool_choice 或 tools 参数)
    以后还要考虑其它的组合（如文件对话）
    返回与 openai 兼容的 Dict
    """
    # import rich
    # rich.print(body)

    # 当调用本接口且 body 中没有传入 "max_tokens" 参数时, 默认使用配置中定义的值
    if body.max_tokens in [None, 0]: # 设置默认最大令牌数
        body.max_tokens = Settings.model_settings.MAX_TOKENS

    client = get_OpenAIClient(model_name=body.model, is_async=True) # 使用 get_OpenAIClient 函数根据提供的模型名称获取异步 OpenAI 客户端
    extra = {**body.model_extra} or {} # 创建一个新的字典 extra，它包含了 body.model_extra 中的所有键值对。or {} 确保了即使 body.model_extra 没有内容，extra 也会是一个空字典。
    for key in list(extra): # 从 body 中删除model_extra里的属性（不删model_extra这个键），以确保 body 只包含必要的字段。
        delattr(body, key)

    # check tools & tool_choice in request body， 如果 tool_choice 或 tools 是字符串，则尝试将其转换为对应的工具对象。
    if isinstance(body.tool_choice, str): # 若body.tool_choice为返字符串形式的工具名称则返回true,否则返回false
        if t := get_tool(body.tool_choice): # := 给变量t赋值后做判断
            body.tool_choice = {"function": {"name": t.name}, "type": "function"} #将 body.tool_choice 重新赋值为一个新的字典，该字典包含工具的名称和类型信息。新字典有一个键 "function"，其值是另一个字典
    if isinstance(body.tools, list): # 若body.tools为返一个工具列表则返回true,否则返回false
        for i in range(len(body.tools)): # 遍历工具列表
            if isinstance(body.tools[i], str): # 若body.tools[i]为字符串形式的工具名称
                if t := get_tool(body.tools[i]): # 若根据给定的工具名称，能在系统中找到对应的工具函数，则为true
                    body.tools[i] = { # 更新 body.tools[i] 为包含工具详细信息的字典
                        "type": "function",
                        "function": {
                            "name": t.name,
                            "description": t.description,
                            "parameters": t.args,
                        },
                    }

    conversation_id = extra.get("conversation_id")

    # chat based on result from one choiced tool
    if body.tool_choice: # 单工具聊天（RAG）
        tool = get_tool(body.tool_choice["function"]["name"]) # 根据工具名称获取工具处理函数
        if not body.tools: # 如果 body.tools 是空的，下面将建一个新列表，并将其赋值给 body.tools，新列表中包含一个字典，该字典描述了一个工具的详细信息。
            body.tools = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.args,
                    },
                }
            ]
        if tool_input := extra.get("tool_input"):# 如果tool_input为真，则进入循环
            try: # 将当前消息保存到数据库
                message_id = (
                    add_message_to_db(
                        chat_type="tool_call",
                        query=body.messages[-1]["content"],
                        conversation_id=conversation_id,
                    )
                    if conversation_id
                    else None
                )
            except Exception as e:
                logger.warning(f"failed to add message to db: {e}")
                message_id = None

            tool_result = await tool.ainvoke(tool_input) # 执行工具调用并返回结果
            prompt_template = PromptTemplate.from_template( # 根据工具的结果构造新的提示模板
                get_prompt_template("rag", "default"), template_format="jinja2"
            )
            body.messages[-1]["content"] = prompt_template.format( # body 是一个字典，其中包含一个键 messages。body.messages 是一个列表，[-1] 表示列表中的最后一个元素。在 Python 中，索引 -1 用来访问列表的最后一个元素，["content"]表示获取的列表中最后一个字典元素的 content 键对应的值
                context=tool_result, question=body.messages[-1]["content"] # 将检索出的文本和问题填充到模板中
            )
            del body.tools # 清除 body 中不再需要的工具相关信息
            del body.tool_choice
            extra_json = { # 构建 extra_json 和 header，准备后续处理
                "message_id": message_id,
                "status": None,
                "model": body.model,
            }
            header = [
                {
                    **extra_json,
                    "content": f"{tool_result}",
                    "tool_call": tool.get_name(),
                    "tool_output": tool_result.data,
                    "is_ref": False if tool.return_direct else True,
                }
            ]
            if tool.return_direct: # 直接返回工具结果：如果工具配置为直接返回结果（return_direct=True），则通过 Server-Sent Events (SSE) 实时返回工具输出
                def temp_gen():
                    yield OpenAIChatOutput(**header[0]).model_dump_json()
                return EventSourceResponse(temp_gen())
            else: # 这里的逻辑可以理解为，把召回的文档和问题经过模板组合后给到大模型生成回复
                return await openai_request(
                    client.chat.completions.create,
                    body,
                    extra_json=extra_json,
                    header=header,
                )

    # agent chat with tool calls
    if body.tools: # 多工具聊天（Agent）
        try:
            message_id = ( # 将当前的消息保存到数据库中，并获取生成的消息 ID
                add_message_to_db(
                    chat_type="agent_chat",
                    query=body.messages[-1]["content"],
                    conversation_id=conversation_id,
                )
                if conversation_id
                else None
            )
        except Exception as e:
            logger.warning(f"failed to add message to db: {e}")
            message_id = None

        chat_model_config = {}  # TODO: 前端支持配置模型
        tool_names = [x["function"]["name"] for x in body.tools if isinstance(x, dict) and "function" in x] # 提取所有工具的名称
        tool_config = {name: get_tool_config(name) for name in tool_names} # 为每个工具获取其配置
        result = await chat( # 异步调用聊天函数
            query=body.messages[-1]["content"], # 从请求体 (body) 中提取最近一条消息的内容，并将其赋值给变量 query。
            metadata=extra.get("metadata", {}),
            conversation_id=extra.get("conversation_id", ""),
            message_id=message_id,
            history_len=-1,
            history=body.messages[:-1], # 从请求体 (body) 中提取所有消息，除了最新的那一条，并将这些消息赋值给变量 history。
            stream=body.stream,
            chat_model_config=extra.get("chat_model_config", chat_model_config), # 从字典 extra 中获取键为 "chat_model_config" 的值，如果 extra 字典中不存在该键，则使用右边的 chat_model_config 变量的值。eg:{"model_name": "gpt-3.5-turbo", "temperature": 0.5, "max_tokens": 1024}
            tool_config=extra.get("tool_config", tool_config),
            max_tokens=body.max_tokens,
        )
        return result
    else:  # LLM chat directly
        try: # query is complex object that unable add to db when using qwen-vl-chat 
            message_id = (
                add_message_to_db(
                    chat_type="llm_chat",
                    query=body.messages[-1]["content"],
                    conversation_id=conversation_id,
                )
                if conversation_id
                else None
            )
        except Exception as e:
            logger.warning(f"failed to add message to db: {e}")
            message_id = None

        extra_json = {
            "message_id": message_id,
            "status": None,
        }
        return await openai_request(
            client.chat.completions.create,
            body,
            extra_json=extra_json
        )
