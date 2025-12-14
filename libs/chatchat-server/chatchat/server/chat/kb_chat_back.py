# '''这个脚本提供了一个接口，用户可以通过它发送查询，然后脚本根据不同的模式使用知识库或搜索引擎来生成聊天响应，并以流式传输或一次性返回给用户。'''
# from __future__ import annotations
#
# import asyncio, json
# import hashlib
# import uuid
# from typing import AsyncIterable, List, Optional, Literal
#
# from fastapi import Body, Request, Header
# from fastapi.concurrency import run_in_threadpool
# from sse_starlette.sse import EventSourceResponse
# from langchain.callbacks import AsyncIteratorCallbackHandler
# from langchain.prompts.chat import ChatPromptTemplate
#
# from chatchat.server.intent_slot.slot_middleware import process_with_slot_model
# from chatchat.settings import Settings
# from chatchat.server.agent.tools_factory.search_internet import search_engine
# from chatchat.server.api_server.api_schemas import OpenAIChatOutput
# from chatchat.server.chat.utils import History
# from chatchat.server.knowledge_base.kb_service.base import KBServiceFactory
# from chatchat.server.knowledge_base.kb_doc_api import search_docs, search_temp_docs
# from chatchat.server.knowledge_base.utils import format_reference
# from chatchat.server.utils import (wrap_done, get_ChatOpenAI, get_default_llm,
#                                    BaseResponse, get_prompt_template, build_logger,
#                                    check_embed_model, api_address
#                                    )
#
# logger = build_logger()
#
#
# async def kb_chat_back(
#         query: str = Body(..., description="用户输入", examples=["你好"]),
#         mode: Literal["local_kb", "temp_kb", "search_engine"] = Body("local_kb", description="知识来源"),
#         kb_name: str = Body("", description="...", examples=["samples"]),
#         top_k: int = Body(Settings.kb_settings.VECTOR_SEARCH_TOP_K, description="匹配向量数"),
#         score_threshold: float = Body(
#             Settings.kb_settings.SCORE_THRESHOLD,
#             description="知识库匹配相关度阈值...",
#             ge=0,
#             le=1,
#         ),
#         history: List[History] = Body(
#             [],
#             description="历史对话",
#             examples=[
#                 [
#                     {"role": "user", "content": "你好"},
#                     {"role": "assistant", "content": "你好！有什么我可以帮您的吗？"}
#                 ]
#             ],
#         ),
#         stream: bool = Body(True, description="流式输出"),
#         model: str = Body(get_default_llm(), description="LLM 模型名称。"),
#         temperature: float = Body(Settings.model_settings.TEMPERATURE, description="LLM 采样温度", ge=0.0, le=1.0),
#         max_tokens: Optional[int] = Body(Settings.model_settings.MAX_TOKENS, description="限制LLM生成Token数量"),
#         prompt_name: str = Body("default", description="使用的prompt模板名称"),
#         return_direct: bool = Body(False, description="直接返回检索结果，不送入 LLM"),
#         conversation_id: str = Header(None, alias="conversation-id"),
#         request: Request = None,
# ):
#     # ✅ 自动生成稳定 conversation_id（当未提供时）
#     if not conversation_id:
#         if request is None:
#             conversation_id = str(uuid.uuid4())
#         else:
#             client_ip = request.client.host if request.client else "unknown"
#             user_agent = request.headers.get("user-agent", "")
#             unique_str = f"{client_ip}:{user_agent}"
#             conversation_id = hashlib.sha256(unique_str.encode()).hexdigest()[:32]
#
#     # ========== 槽位预处理 ==========
#     is_final_response, content, meta = await process_with_slot_model(query, conversation_id)
#
#     # 判断是否需要走 RAG 流程
#     should_run_rag = False
#
#     if is_final_response:
#         if meta == "[SLOT_COMPLETED]":
#             # 槽位已完成，使用 content 作为标准化查询去 RAG
#             query = content
#             should_run_rag = True
#         elif meta in ("[GREETING]", "[CANCEL]", "[ERROR]"):
#             # 系统主动回复，直接返回给用户
#             output = OpenAIChatOutput(
#                 id=f"chat{uuid.uuid4()}",
#                 object="chat.completion" if not stream else "chat.completion.chunk",
#                 content=content,
#                 role="assistant",
#                 model=model,
#                 docs=[],
#             )
#             if stream:
#                 return EventSourceResponse([output.model_dump_json()])
#             else:
#                 return output.model_dump_json()
#         else:
#             # 兜底：其他 is_final_response 场景也直接返回
#             output = OpenAIChatOutput(
#                 id=f"chat{uuid.uuid4()}",
#                 object="chat.completion" if not stream else "chat.completion.chunk",
#                 content=content,
#                 role="assistant",
#                 model=model,
#                 docs=[],
#             )
#             if stream:
#                 return EventSourceResponse([output.model_dump_json()])
#             else:
#                 return output.model_dump_json()
#     else:
#         # is_final_response == False
#         if meta == "[NEED_MORE]":
#             # 填槽中，返回追问语句
#             output = OpenAIChatOutput(
#                 id=f"chat{uuid.uuid4()}",
#                 object="chat.completion" if not stream else "chat.completion.chunk",
#                 content=content,
#                 role="assistant",
#                 model=model,
#                 docs=[],
#             )
#             if stream:
#                 return EventSourceResponse([output.model_dump_json()])
#             else:
#                 return output.model_dump_json()
#         elif meta == "[NO_SCENE]":
#             # 未命中场景，用原始 query 走 RAG
#             query = content
#             should_run_rag = True
#         else:
#             # 安全兜底：当作普通查询
#             query = content
#             should_run_rag = True
#
#     # ========== 安全校验：防止空 query 进 RAG ==========
#     if not query or not query.strip():
#         output = OpenAIChatOutput(
#             id=f"chat{uuid.uuid4()}",
#             object="chat.completion" if not stream else "chat.completion.chunk",
#             content="请输入有效问题。",
#             role="assistant",
#             model=model,
#             docs=[],
#         )
#         if stream:
#             return EventSourceResponse([output.model_dump_json()])
#         else:
#             return output.model_dump_json()
#     # todo =============================== 槽位预处理 ===================================
#
#     if mode == "local_kb":
#         kb = KBServiceFactory.get_service_by_name(kb_name)
#         if kb is None:
#             return BaseResponse(code=404, msg=f"未找到知识库 {kb_name}")
#
#     async def knowledge_base_chat_iterator() -> AsyncIterable[str]:
#         try:
#             nonlocal history, prompt_name, max_tokens
#
#             history = [History.from_data(h) for h in history]
#
#             if mode == "local_kb":
#                 kb = KBServiceFactory.get_service_by_name(kb_name)
#                 ok, msg = kb.check_embed_model()
#                 if not ok:
#                     raise ValueError(msg)
#                 docs = await run_in_threadpool(search_docs,
#                                                query=query,
#                                                knowledge_base_name=kb_name,
#                                                top_k=top_k,
#                                                score_threshold=score_threshold,
#                                                file_name="",
#                                                metadata={})
#                 source_documents = format_reference(kb_name, docs, api_address(is_public=True))
#             elif mode == "temp_kb":
#                 ok, msg = check_embed_model()
#                 if not ok:
#                     raise ValueError(msg)
#                 docs = await run_in_threadpool(search_temp_docs,
#                                                kb_name,
#                                                query=query,
#                                                top_k=top_k,
#                                                score_threshold=score_threshold)
#                 source_documents = format_reference(kb_name, docs, api_address(is_public=True))
#             elif mode == "search_engine":
#                 result = await run_in_threadpool(search_engine, query, top_k, kb_name)
#                 docs = [x.dict() for x in result.get("docs", [])]
#                 source_documents = [
#                     f"""出处 [{i + 1}] [{d['metadata']['filename']}]({d['metadata']['source']}) \n\n{d['page_content']}\n\n"""
#                     for i, d in enumerate(docs)]
#             else:
#                 docs = []
#                 source_documents = []
#             # import rich
#             # rich.print(dict(
#             #     mode=mode,
#             #     query=query,
#             #     knowledge_base_name=kb_name,
#             #     top_k=top_k,
#             #     score_threshold=score_threshold,
#             # ))
#             # rich.print(docs)
#             if return_direct:
#                 yield OpenAIChatOutput(
#                     id=f"chat{uuid.uuid4()}",
#                     model=None,
#                     object="chat.completion",
#                     content="",
#                     role="assistant",
#                     finish_reason="stop",
#                     docs=source_documents,
#                 ).model_dump_json()
#                 return
#
#             callback = AsyncIteratorCallbackHandler()
#             callbacks = [callback]
#
#             # Enable langchain-chatchat to support langfuse
#             import os
#             langfuse_secret_key = os.environ.get('LANGFUSE_SECRET_KEY')
#             langfuse_public_key = os.environ.get('LANGFUSE_PUBLIC_KEY')
#             langfuse_host = os.environ.get('LANGFUSE_HOST')
#             if langfuse_secret_key and langfuse_public_key and langfuse_host:
#                 from langfuse import Langfuse
#                 from langfuse.callback import CallbackHandler
#                 langfuse_handler = CallbackHandler()
#                 callbacks.append(langfuse_handler)
#
#             if max_tokens in [None, 0]:
#                 max_tokens = Settings.model_settings.MAX_TOKENS
#
#             llm = get_ChatOpenAI(
#                 model_name=model,
#                 temperature=temperature,
#                 max_tokens=max_tokens,
#                 callbacks=callbacks,
#             )
#             # TODO： 视情况使用 API
#             # # 加入reranker
#             # if Settings.kb_settings.USE_RERANKER:
#             #     reranker_model_path = get_model_path(Settings.kb_settings.RERANKER_MODEL)
#             #     reranker_model = LangchainReranker(top_n=top_k,
#             #                                     device=embedding_device(),
#             #                                     max_length=Settings.kb_settings.RERANKER_MAX_LENGTH,
#             #                                     model_name_or_path=reranker_model_path
#             #                                     )
#             #     print("-------------before rerank-----------------")
#             #     print(docs)
#             #     docs = reranker_model.compress_documents(documents=docs,
#             #                                              query=query)
#             #     print("------------after rerank------------------")
#             #     print(docs)
#             context = "\n\n".join([doc["page_content"] for doc in docs])
#
#             if len(docs) == 0:  # 如果没有找到相关文档，使用empty模板
#                 prompt_name = "empty"
#             prompt_template = get_prompt_template("rag", prompt_name)
#             input_msg = History(role="user", content=prompt_template).to_msg_template(False)
#             chat_prompt = ChatPromptTemplate.from_messages(
#                 [i.to_msg_template() for i in history] + [input_msg])
#
#             chain = chat_prompt | llm
#
#             # Begin a task that runs in the background.
#             task = asyncio.create_task(wrap_done(
#                 chain.ainvoke({"context": context, "question": query}),
#                 callback.done),
#             )
#
#             if len(source_documents) == 0:  # 没有找到相关文档
#                 source_documents.append(f"<span style='color:red'>未找到相关文档,该回答为大模型自身能力解答！</span>")
#
#             if stream:
#                 # yield documents first
#                 ret = OpenAIChatOutput(
#                     id=f"chat{uuid.uuid4()}",
#                     object="chat.completion.chunk",
#                     content="",
#                     role="assistant",
#                     model=model,
#                     docs=source_documents,
#                 )
#                 yield ret.model_dump_json()
#
#                 async for token in callback.aiter():
#                     ret = OpenAIChatOutput(
#                         id=f"chat{uuid.uuid4()}",
#                         object="chat.completion.chunk",
#                         content=token,
#                         role="assistant",
#                         model=model,
#                     )
#                     yield ret.model_dump_json()
#             else:
#                 answer = ""
#                 async for token in callback.aiter():
#                     answer += token
#                 ret = OpenAIChatOutput(
#                     id=f"chat{uuid.uuid4()}",
#                     object="chat.completion",
#                     content=answer,
#                     role="assistant",
#                     model=model,
#                 )
#                 yield ret.model_dump_json()
#             await task
#         except asyncio.exceptions.CancelledError:
#             logger.warning("streaming progress has been interrupted by user.")
#             return
#         except Exception as e:
#             logger.error(f"error in knowledge chat: {e}")
#             yield {"data": json.dumps({"error": str(e)})}
#             return
#
#     if stream:
#         return EventSourceResponse(knowledge_base_chat_iterator())
#     else:
#         return await knowledge_base_chat_iterator().__anext__()
