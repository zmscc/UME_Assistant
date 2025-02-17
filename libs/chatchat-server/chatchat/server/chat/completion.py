import asyncio
from typing import AsyncIterable, Optional

from fastapi import Body
from langchain.callbacks import AsyncIteratorCallbackHandler
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from sse_starlette.sse import EventSourceResponse

from chatchat.server.utils import get_OpenAI, get_prompt_template, wrap_done, build_logger


logger = build_logger()

'''completion 方法是本项目中某些 FastAPI 应用程序中用于处理文本补全请求的核心逻辑。它接受用户输入的查询，并根据配置的参数，使用指定的语言模型（LLM）生成响应'''
async def completion( # 通过 FastAPI 的 Body 参数，completion 接收来自客户端的 POST 请求体中的各项参数（这些参数是从调用completion函数的请求体中解析出来的）
    query: str = Body(..., description="用户输入", examples=["恼羞成怒"]),
    stream: bool = Body(False, description="流式输出"),
    echo: bool = Body(False, description="除了输出之外，还回显输入"),
    model_name: str = Body(None, description="LLM 模型名称。"),
    temperature: float = Body(0.01, description="LLM 采样温度", ge=0.0, le=1.0),
    max_tokens: Optional[int] = Body(
        1024, description="限制LLM生成Token数量，默认None代表模型最大值"
    ),
    # top_p: float = Body(TOP_P, description="LLM 核采样。勿与temperature同时设置", gt=0.0, lt=1.0),
    prompt_name: str = Body(
        "default", description="使用的prompt模板名称(在configs/prompt_config.py中配置)"
    ),
):
    '''这个内部函数负责实际的文本补全任务。它创建了一个异步迭代器，该迭代器可以逐块地产生补全结果。'''
    # TODO: 因ApiModelWorker 默认是按chat处理的，会对params["prompt"] 解析为messages，因此ApiModelWorker 使用时需要有相应处理
    async def completion_iterator(
        query: str,
        model_name: str = None,
        prompt_name: str = prompt_name,
        echo: bool = echo,
    ) -> AsyncIterable[str]:
        try:
            nonlocal max_tokens
            callback = AsyncIteratorCallbackHandler() # 它首先设置回调处理器 AsyncIteratorCallbackHandler，这是为了能够捕获 LLM 生成的每个部分的结果，并在流式模式下实时发送给客户端。
            if isinstance(max_tokens, int) and max_tokens <= 0: # 当 max_tokens 同时满足是整数且小于或等于0时,则将 max_tokens 的值设置为 None
                max_tokens = None

            model = get_OpenAI( # 初始化语言模型
                model_name=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                callbacks=[callback],
                echo=echo,
                local_wrap=True,
            )

            prompt_template = get_prompt_template("llm_model", prompt_name) # 从预定义的提示模板中加载一个特定的模板
            prompt = PromptTemplate.from_template(prompt_template, template_format="jinja2") # 构建一个 PromptTemplate 对象
            chain = LLMChain(prompt=prompt, llm=model) # 使用提示模板和语言模型创建一个 LLMChain 实例，这个链对象用来执行实际的推理过程。

            # Begin a task that runs in the background.
            task = asyncio.create_task(
                # 传入 {"input": query} 的原因是由于 LLMChain（或类似的链式处理结构）期望接收一个字典形式的输入，其中键名对应于模板中定义的变量名。
                wrap_done(chain.acall({"input": query}), callback.done), # 调用 chain.acall() 方法开始一次推理请求，并通过 asyncio.create_task() 将其作为后台任务运行。这样可以确保即使在长时间生成过程中也不会影响其他操作。
            )

            if stream: # 如果启用了流式输出，则直接通过异步迭代返回生成的每个令牌；否则，收集所有令牌并一次性返回完整的答案。
                async for token in callback.aiter():
                    # Use server-sent-events to stream the response
                    yield token
            else:
                answer = ""
                async for token in callback.aiter():
                    answer += token
                yield answer

            await task
        except asyncio.exceptions.CancelledError:
            logger.warning("streaming progress has been interrupted by user.")
            return

    '''
    所有生成的内容都会被打包成一个 EventSourceResponse，这是一种特别适合流式传输数据的 HTTP 响应类型。
    客户端可以通过监听服务器发送事件（SSE）来接收实时更新的文本块。
    '''
    return EventSourceResponse( # EventSourceResponse 是一种允许服务器向客户端推送更新的技术
        completion_iterator( # 将 completion_iterator 函数作为参数传递进去。这并不是直接调用 completion_iterator，而是将它作为一个可迭代的对象（具体来说是一个异步生成器）传入。
            query=query, model_name=model_name, prompt_name=prompt_name
        ), # 当 FastAPI 准备发送 HTTP 响应时，它会自动开始遍历这个异步生成器，逐个读取并发送每个生成的结果给客户端。
    )
