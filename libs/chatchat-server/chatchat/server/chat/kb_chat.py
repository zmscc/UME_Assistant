'''这个脚本提供了一个接口，用户可以通过它发送查询，然后脚本根据不同的模式使用知识库或搜索引擎来生成聊天响应，并以流式传输或一次性返回给用户。'''
from __future__ import annotations

import asyncio, json
import hashlib
import uuid
from typing import AsyncIterable, List, Optional, Literal

from fastapi import Body, Request, Header
from fastapi.concurrency import run_in_threadpool
from sse_starlette.sse import EventSourceResponse
from langchain.callbacks import AsyncIteratorCallbackHandler
from langchain.prompts.chat import ChatPromptTemplate

from chatchat.server.intent_slot.slot_middleware import process_with_slot_model
from chatchat.settings import Settings
from chatchat.server.agent.tools_factory.search_internet import search_engine
from chatchat.server.api_server.api_schemas import OpenAIChatOutput
from chatchat.server.chat.utils import History
from chatchat.server.knowledge_base.kb_service.base import KBServiceFactory
from chatchat.server.knowledge_base.kb_doc_api import search_docs, search_temp_docs
from chatchat.server.knowledge_base.utils import format_reference
from chatchat.server.utils import (wrap_done, get_ChatOpenAI, get_default_llm,
                                   BaseResponse, get_prompt_template, build_logger,
                                   check_embed_model, api_address
                                )


logger = build_logger()


async def kb_chat(
    query: str = Body(..., description="用户输入", examples=["你好"]),
    mode: Literal["local_kb", "temp_kb", "search_engine"] = Body("local_kb", description="知识来源"),
    kb_name: str = Body("", description="...", examples=["samples"]),
    top_k: int = Body(Settings.kb_settings.VECTOR_SEARCH_TOP_K, description="匹配向量数"),
    score_threshold: float = Body(
        Settings.kb_settings.SCORE_THRESHOLD,
        description="知识库匹配相关度阈值...",
        ge=0,
        le=1,
    ),
    history: List[History] = Body(
        [],
        description="历史对话",
        examples=[
            [
                {"role": "user", "content": "你好"},
                {"role": "assistant", "content": "你好！有什么我可以帮您的吗？"}
            ]
        ],
    ),
    stream: bool = Body(True, description="流式输出"),
    model: str = Body(get_default_llm(), description="LLM 模型名称。"),
    temperature: float = Body(Settings.model_settings.TEMPERATURE, description="LLM 采样温度", ge=0.0, le=1.0),
    max_tokens: Optional[int] = Body(Settings.model_settings.MAX_TOKENS, description="限制LLM生成Token数量"),
    prompt_name: str = Body("default", description="使用的prompt模板名称"),
    return_direct: bool = Body(False, description="直接返回检索结果，不送入 LLM"),
    conversation_id: str = Header(None, alias="conversation-id"),
    request: Request = None,
):
    # 自动生成稳定 conversation_id（当未提供时）
    if not conversation_id:
        if request is None:
            conversation_id = str(uuid.uuid4())
        else:
            client_ip = request.client.host if request.client else "unknown"
            user_agent = request.headers.get("user-agent", "")
            unique_str = f"{client_ip}:{user_agent}"
            conversation_id = hashlib.sha256(unique_str.encode()).hexdigest()[:32]

    # ========== 槽位预处理（唯一允许修改的部分）==========
    is_final_response, content, meta = await process_with_slot_model(query, conversation_id)

    # 初始化控制变量（保持原名）
    system_reply: Optional[str] = None
    should_run_rag = False

    # ✅ 仅修改以下逻辑块（根据 meta 精确路由）
    if meta == "[SLOT_COMPLETED]":
        # 槽位已完成：使用 content 作为最终 query，走完整 RAG+LLM 流程
        query = content
        should_run_rag = True
        system_reply = None  # 关键：不设系统回复

    elif meta == "[NEED_MORE]":
        # 需要更多信息：返回追问语句，不走 RAG
        system_reply = content
        should_run_rag = False

    elif meta in ("[GREETING]", "[CANCEL]", "[ERROR]"):
        # 系统主动消息：直接回复
        system_reply = content
        should_run_rag = False

    elif meta == "[NO_SCENE]":
        # 未命中场景：尝试用 content 走 RAG
        query = content
        should_run_rag = True
        system_reply = None

    else:
        # 未知 meta：保守当作普通查询
        query = content if content.strip() else query
        should_run_rag = True
        system_reply = None

    # 安全校验：防止空 query
    if not query or not query.strip():
        system_reply = "请输入有效问题。"
        should_run_rag = False

    # ========== 以下代码完全不变（包括 knowledge_base_chat_iterator）==========

    # 知识库服务校验（仅在需要时）
    if should_run_rag and mode == "local_kb":
        kb = KBServiceFactory.get_service_by_name(kb_name)
        if kb is None:
            return BaseResponse(code=404, msg=f"未找到知识库 {kb_name}")

    async def knowledge_base_chat_iterator() -> AsyncIterable[str]:
        try:
            nonlocal history, prompt_name, max_tokens, system_reply, should_run_rag

            history = [History.from_data(h) for h in history]

            # 统一处理系统回复路径
            if system_reply is not None:
                source_documents = ["<span style='color:gray'>[系统消息]</span>"]

                if stream:
                    ret = OpenAIChatOutput(
                        id=f"chat{uuid.uuid4()}",
                        object="chat.completion.chunk",
                        content="",
                        role="assistant",
                        model=model,
                        docs=source_documents,
                    )
                    yield ret.model_dump_json()

                    ret = OpenAIChatOutput(
                        id=f"chat{uuid.uuid4()}",
                        object="chat.completion.chunk",
                        content=system_reply,
                        role="assistant",
                        model=model,
                    )
                    yield ret.model_dump_json()
                else:
                    ret = OpenAIChatOutput(
                        id=f"chat{uuid.uuid4()}",
                        object="chat.completion",
                        content=system_reply,
                        role="assistant",
                        model=model,
                        docs=source_documents,
                    )
                    yield ret.model_dump_json()
                return

            # ========== 以下为原始 RAG + LLM 逻辑（完全未改动） ==========
            if mode == "local_kb":
                kb = KBServiceFactory.get_service_by_name(kb_name)
                ok, msg = kb.check_embed_model()
                if not ok:
                    raise ValueError(msg)
                docs = await run_in_threadpool(
                    search_docs,
                    query=query,
                    knowledge_base_name=kb_name,
                    top_k=top_k,
                    score_threshold=score_threshold,
                    file_name="",
                    metadata={},
                )
                source_documents = format_reference(kb_name, docs, api_address(is_public=True))
            elif mode == "temp_kb":
                ok, msg = check_embed_model()
                if not ok:
                    raise ValueError(msg)
                docs = await run_in_threadpool(
                    search_temp_docs,
                    kb_name,
                    query=query,
                    top_k=top_k,
                    score_threshold=score_threshold,
                )
                source_documents = format_reference(kb_name, docs, api_address(is_public=True))
            elif mode == "search_engine":
                result = await run_in_threadpool(search_engine, query, top_k, kb_name)
                docs = [x.dict() for x in result.get("docs", [])]
                source_documents = [
                    f"""出处 [{i + 1}] [{d['metadata']['filename']}]({d['metadata']['source']}) \n\n{d['page_content']}\n\n"""
                    for i, d in enumerate(docs)
                ]
            else:
                docs = []
                source_documents = []

            if return_direct:
                yield OpenAIChatOutput(
                    id=f"chat{uuid.uuid4()}",
                    model=None,
                    object="chat.completion",
                    content="",
                    role="assistant",
                    finish_reason="stop",
                    docs=source_documents,
                ).model_dump_json()
                return

            callback = AsyncIteratorCallbackHandler()
            callbacks = [callback]

            import os
            langfuse_secret_key = os.environ.get('LANGFUSE_SECRET_KEY')
            langfuse_public_key = os.environ.get('LANGFUSE_PUBLIC_KEY')
            langfuse_host = os.environ.get('LANGFUSE_HOST')
            if langfuse_secret_key and langfuse_public_key and langfuse_host:
                from langfuse.callback import CallbackHandler
                callbacks.append(CallbackHandler())

            if max_tokens in [None, 0]:
                max_tokens = Settings.model_settings.MAX_TOKENS

            llm = get_ChatOpenAI(
                model_name=model,
                temperature=temperature,
                max_tokens=max_tokens,
                callbacks=callbacks,
            )

            context = "\n\n".join([doc["page_content"] for doc in docs])

            if len(docs) == 0:
                prompt_name = "empty"

            prompt_template = get_prompt_template("rag", prompt_name)
            input_msg = History(role="user", content=prompt_template).to_msg_template(False)
            chat_prompt = ChatPromptTemplate.from_messages(
                [i.to_msg_template() for i in history] + [input_msg]
            )

            chain = chat_prompt | llm

            task = asyncio.create_task(wrap_done(
                chain.ainvoke({"context": context, "question": query}),
                callback.done),
            )

            if len(source_documents) == 0:
                source_documents.append("<span style='color:red'>未找到相关文档,该回答为大模型自身能力解答！</span>")

            if stream:
                ret = OpenAIChatOutput(
                    id=f"chat{uuid.uuid4()}",
                    object="chat.completion.chunk",
                    content="",
                    role="assistant",
                    model=model,
                    docs=source_documents,
                )
                yield ret.model_dump_json()

                async for token in callback.aiter():
                    ret = OpenAIChatOutput(
                        id=f"chat{uuid.uuid4()}",
                        object="chat.completion.chunk",
                        content=token,
                        role="assistant",
                        model=model,
                    )
                    yield ret.model_dump_json()
            else:
                answer = ""
                async for token in callback.aiter():
                    answer += token
                ret = OpenAIChatOutput(
                    id=f"chat{uuid.uuid4()}",
                    object="chat.completion",
                    content=answer,
                    role="assistant",
                    model=model,
                    docs=source_documents,
                )
                yield ret.model_dump_json()

            await task

        except asyncio.exceptions.CancelledError:
            logger.warning("streaming progress has been interrupted by user.")
            return
        except Exception as e:
            logger.error(f"error in knowledge chat: {e}")
            yield json.dumps({"error": str(e)})
            return

    if stream:
        return EventSourceResponse(knowledge_base_chat_iterator())
    else:
        return await knowledge_base_chat_iterator().__anext__()