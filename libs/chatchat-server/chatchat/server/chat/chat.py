'''这个脚本提供了一个接口，用户可以通过它发送消息，然后脚本会使用聊天模型生成响应，并将这些响应以流式传输或一次性返回给用户。'''
import asyncio
import json
import uuid
from typing import AsyncIterable, List

from fastapi import Body
from langchain.chains import LLMChain
from langchain.prompts.chat import ChatPromptTemplate
from langchain_core.messages import AIMessage, HumanMessage, convert_to_messages
from sse_starlette.sse import EventSourceResponse

from chatchat.settings import Settings
from chatchat.server.agent.agent_factory.agents_registry import agents_registry
from chatchat.server.api_server.api_schemas import OpenAIChatOutput
from chatchat.server.callback_handler.agent_callback_handler import (
    AgentExecutorAsyncIteratorCallbackHandler,
    AgentStatus,
)
from chatchat.server.chat.utils import History
from chatchat.server.memory.conversation_db_buffer_memory import (
    ConversationBufferDBMemory,
)
from chatchat.server.utils import (
    MsgType,
    get_ChatOpenAI,
    get_prompt_template,
    get_tool,
    wrap_done,
    get_default_llm,
    build_logger,
)


logger = build_logger()


'''此函数根据配置创建和返回模型实例和提示模板。'''
def create_models_from_config(configs, callbacks, stream, max_tokens): # todo 这个方法还没读透彻
    configs = configs or Settings.model_settings.LLM_MODEL_CONFIG # 如果没有configs被赋值（即 configs 是 None 或者没有被定义），那么它会将 LLM_MODEL_CONFIG 的值赋给 configs。
    models = {}
    prompts = {}
    for model_type, params in configs.items():
        model_name = params.get("model", "").strip() or get_default_llm() # 如果params有"model"键则返回对应的值，否则返回""；strip()移除字符串开头和结尾的空白字符；如果前面的表达式返回的是""（即假值），则调用get_default_llm()
        callbacks = callbacks if params.get("callbacks", False) else None
        # 判断是否传入 max_tokens 的值, 如果传入就按传入的赋值(api 调用且赋值), 如果没有传入则按照初始化配置赋值(ui 调用或 api 调用未赋值)
        max_tokens_value = max_tokens if max_tokens is not None else params.get("max_tokens", 1000)
        model_instance = get_ChatOpenAI(
            model_name=model_name,
            temperature=params.get("temperature", 0.5),
            max_tokens=max_tokens_value,
            callbacks=callbacks,
            streaming=stream,
            local_wrap=True,
        )
        models[model_type] = model_instance # 将model_instance作为值，model_type作为键，存入models字典中
        prompt_name = params.get("prompt_name", "default")
        prompt_template = get_prompt_template(type=model_type, name=prompt_name)
        prompts[model_type] = prompt_template
    return models, prompts


'''此函数根据提供的参数创建聊天链（LLMChain）或代理执行器（AgentExecutor），用于处理聊天对话。'''
def create_models_chains(
    history, history_len, prompts, models, tools, callbacks, conversation_id, metadata # callbacks参数: 回调处理器列表，用于监控和扩展功能
):
    memory = None # 用于存储对话历史的内存对象
    chat_prompt = None # 将要使用的聊天提示模板

    if history: # 如果有历史记录
        history = [History.from_data(h) for h in history] # 将历史记录转换为 History 对象列表。
        input_msg = History(role="user", content=prompts["llm_model"]).to_msg_template( # 创建一个新的用户输入消息模板
            False
        )
        chat_prompt = ChatPromptTemplate.from_messages( # 使用历史记录和新输入消息创建 ChatPromptTemplate。
            [i.to_msg_template() for i in history] + [input_msg]
        )
    elif conversation_id and history_len > 0: # 如果有对话 ID 和历史记录长度限制：
        memory = ConversationBufferDBMemory( # 创建一个 ConversationBufferDBMemory 对象，用于存储对话历史，限制消息数量为 history_len
            conversation_id=conversation_id,
            llm=models["llm_model"],
            message_limit=history_len,
        )
    else: # 直接创建一个只包含新输入消息的 ChatPromptTemplate
        input_msg = History(role="user", content=prompts["llm_model"]).to_msg_template(
            False
        )
        chat_prompt = ChatPromptTemplate.from_messages([input_msg])

    if "action_model" in models and tools:
        llm = models["action_model"]
        llm.callbacks = callbacks # 使用 action_model 作为语言模型，并为其添加回调处理器。
        agent_executor = agents_registry( # 使用 agents_registry 函数创建一个agent执行器 (agent_executor)，它结合了语言模型、工具和回调处理器。
            llm=llm, callbacks=callbacks, tools=tools, prompt=None, verbose=True
        )
        full_chain = {"input": lambda x: x["input"]} | agent_executor # 构建一个处理链，先提取输入，然后通过代理执行器处理
    else:
        llm = models["llm_model"] # 使用默认的语言模型 (llm_model)，并为其添加回调处理器。
        llm.callbacks = callbacks
        chain = LLMChain(prompt=chat_prompt, llm=llm, memory=memory) # 创建一个 LLMChain，结合聊天提示模板、语言模型和内存对象（如果有的话）。
        full_chain = {"input": lambda x: x["input"]} | chain # 构建一个处理链，首先提取输入，然后通过 LLMChain 处理
    return full_chain


'''
这是脚本的主要部分，它定义了一个异步的 FastAPI 路由处理函数。
它接受多个参数，包括用户输入、元数据、会话 ID、消息 ID、历史对话长度和内容等。
函数内部定义了一个异步生成器 chat_iterator，它用于流式传输聊天机器人的响应。
如果 stream 参数为 True，则使用 EventSourceResponse 返回一个流式响应；否则，它会收集所有的响应并返回一个完成的响应。
'''
async def chat(
    query: str = Body(..., description="用户输入", examples=["恼羞成怒"]), # ... 表示这是一个必填项，即客户端必须提供这个参数，Body(...) 用于明确指出该参数是从请求体中获取的
    metadata: dict = Body({}, description="附件，可能是图像或者其他功能", examples=[]),
    conversation_id: str = Body("", description="对话框ID"),
    message_id: str = Body(None, description="数据库消息ID"),
    history_len: int = Body(-1, description="从数据库中取历史消息的数量"),
    history: List[History] = Body(
        [],
        description="历史对话，设为一个整数可以从数据库中读取历史消息",
        examples=[
            [
                {"role": "user", "content": "我们来玩成语接龙，我先来，生龙活虎"},
                {"role": "assistant", "content": "虎头虎脑"},
            ]
        ],
    ),
    stream: bool = Body(True, description="流式输出"),
    chat_model_config: dict = Body({}, description="LLM 模型配置", examples=[]),
    tool_config: dict = Body({}, description="工具配置", examples=[]),
    max_tokens: int = Body(None, description="LLM最大token数配置", example=4096),
):
    """Agent 对话"""

    async def chat_iterator() -> AsyncIterable[OpenAIChatOutput]: #该返回一个异步可迭代对象（AsyncIterable），异步迭代器特别适用于处理需要逐步生成数据的场景。该对象生成 OpenAIChatOutput 类型的项。
        try:
            callback = AgentExecutorAsyncIteratorCallbackHandler()
            callbacks = [callback] # 创建callbacks列表

            # Enable langchain-chatchat to support langfuse
            import os

            # langfuse_secret_key = os.environ.get("LANGFUSE_SECRET_KEY")
            # langfuse_public_key = os.environ.get("LANGFUSE_PUBLIC_KEY")
            # langfuse_host = os.environ.get("LANGFUSE_HOST")
            # langfuse_secret_key = "sk-lf-dbbef8f9-4bde-4ddd-9c38-23e9563411f3"
            # langfuse_public_key = "pk-lf-d68680ea-7633-40d6-9892-782a8a0c62f5"
            # langfuse_host = "https://cloud.langfuse.com"
            # if langfuse_secret_key and langfuse_public_key and langfuse_host:
            #     from langfuse import Langfuse
            #     from langfuse.callback import CallbackHandler
            #
            #     langfuse_handler = CallbackHandler()
            #     callbacks.append(langfuse_handler)

            models, prompts = create_models_from_config( # 根据提供的配置创建模型和提示
                callbacks=callbacks, configs=chat_model_config, stream=stream, max_tokens=max_tokens
            )
            all_tools = get_tool().values()
            tools = [tool for tool in all_tools if tool.name in tool_config]
            tools = [t.copy(update={"callbacks": callbacks}) for t in tools]
            full_chain = create_models_chains( # 根据给定的参数创建一个完整的链式处理逻辑,包括提示、模型、对话 ID、工具、回调、历史消息等。
                prompts=prompts,
                models=models,
                conversation_id=conversation_id,
                tools=tools,
                callbacks=callbacks,
                history=history,
                history_len=history_len,
                metadata=metadata,
            )

            _history = [History.from_data(h) for h in history] # 将历史消息转换为适合处理的格式
            chat_history = [h.to_msg_tuple() for h in _history]

            history_message = convert_to_messages(chat_history) # 转换消息类型

            task = asyncio.create_task( # 使用 asyncio.create_task 创建一个异步任务，调用 full_chain.ainvoke 并传递用户输入和历史消息。
                wrap_done(
                    full_chain.ainvoke(
                        {
                            "input": query,
                            "chat_history": history_message,
                        }
                    ),
                    callback.done,
                )
            )

            last_tool = {}
            async for chunk in callback.aiter(): # 通过 callback.aiter() 异步迭代回调数据，处理不同状态下的响应（如工具开始、工具结束、代理完成）。
                data = json.loads(chunk)
                data["tool_calls"] = []
                data["message_type"] = MsgType.TEXT

                if data["status"] == AgentStatus.tool_start:
                    last_tool = {
                        "index": 0,
                        "id": data["run_id"],
                        "type": "function",
                        "function": {
                            "name": data["tool"],
                            "arguments": data["tool_input"],
                        },
                        "tool_output": None,
                        "is_error": False,
                    }
                    data["tool_calls"].append(last_tool)
                if data["status"] in [AgentStatus.tool_end]:
                    last_tool.update(
                        tool_output=data["tool_output"],
                        is_error=data.get("is_error", False),
                    )
                    data["tool_calls"] = [last_tool]
                    last_tool = {}
                    try:
                        tool_output = json.loads(data["tool_output"])
                        if message_type := tool_output.get("message_type"):
                            data["message_type"] = message_type
                    except:
                        ...
                elif data["status"] == AgentStatus.agent_finish:
                    try:
                        tool_output = json.loads(data["text"])
                        if message_type := tool_output.get("message_type"):
                            data["message_type"] = message_type
                    except:
                        ...
                text_value = data.get("text", "")
                content = text_value if isinstance(text_value, str) else str(text_value)
                ret = OpenAIChatOutput(
                    id=f"chat{uuid.uuid4()}",
                    object="chat.completion.chunk",
                    content=content,
                    role="assistant",
                    tool_calls=data["tool_calls"],
                    model=models["llm_model"].model_name,
                    status=data["status"],
                    message_type=data["message_type"],
                    message_id=message_id,
                )
                res = ret.model_dump_json()
                yield res
            # yield OpenAIChatOutput( # return blank text lastly
            #         id=f"chat{uuid.uuid4()}",
            #         object="chat.completion.chunk",
            #         content="",
            #         role="assistant",
            #         model=models["llm_model"].model_name,
            #         status = data["status"],
            #         message_type = data["message_type"],
            #         message_id=message_id,
            # )
            await task
        except asyncio.exceptions.CancelledError:
            logger.warning("streaming progress has been interrupted by user.")
            return
        except Exception as e:
            logger.error(f"error in chat: {e}")
            yield {"data": json.dumps({"error": str(e)})}
            return

    if stream:
        return EventSourceResponse(chat_iterator())
    else:
        ret = OpenAIChatOutput(
            id=f"chat{uuid.uuid4()}",
            object="chat.completion",
            content="",
            role="assistant",
            finish_reason="stop",
            tool_calls=[],
            status=AgentStatus.agent_finish,
            message_type=MsgType.TEXT,
            message_id=message_id,
        )

        async for chunk in chat_iterator():
            data = json.loads(chunk)
            if text := data["choices"][0]["delta"]["content"]:
                ret.content += text
            if data["status"] == AgentStatus.tool_end:
                ret.tool_calls += data["choices"][0]["delta"]["tool_calls"]
            ret.model = data["model"]
            ret.created = data["created"]

        return ret.model_dump()
