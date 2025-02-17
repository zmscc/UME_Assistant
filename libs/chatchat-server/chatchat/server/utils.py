import asyncio
import multiprocessing as mp
import os
import requests
import socket
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlparse
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    Generator,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
)

import httpx
import openai
from fastapi import FastAPI
from langchain.tools import BaseTool
from langchain_core.embeddings import Embeddings
from langchain_openai.chat_models import ChatOpenAI
from langchain_openai.llms import OpenAI
from memoization import cached, CachingAlgorithmFlag

from chatchat.settings import Settings, XF_MODELS_TYPES
from chatchat.server.pydantic_v2 import BaseModel, Field
from chatchat.utils import build_logger
import requests

logger = build_logger()


async def wrap_done(fn: Awaitable, event: asyncio.Event):
    """Wrap an awaitable with a event to signal when it's done or an exception is raised."""
    try:
        await fn
    except Exception as e:
        msg = f"Caught exception: {e}"
        logger.error(f"{e.__class__.__name__}: {msg}")
    finally:
        # Signal the aiter to stop.
        event.set()


'''这个函数接受一个 URL 字符串，解析它以提取方案和网络位置，然后构造并返回一个基础 URL，末尾不包含斜杠。例如，对于输入 “http://127.0.0.1:9997/v1”，它返回 “http://127.0.0.1:9997”'''
def get_base_url(url):
    parsed_url = urlparse(url)  # 解析url
    base_url = '{uri.scheme}://{uri.netloc}/'.format(uri=parsed_url)  # 格式化基础url
    return base_url.rstrip('/')


def get_config_platforms() -> Dict[str, Dict]:
    """
    获取配置的模型平台，会将 pydantic model 转换为字典。
    """
    platforms = [m.model_dump() for m in Settings.model_settings.MODEL_PLATFORMS] # 遍历 Settings.model_settings.MODEL_PLATFORMS 列表中的每个模型平台配置，调用 model_dump() 方法将其转换为字典，并将结果存储在一个新的列表 platforms 中。
    return {m["platform_name"]: m for m in platforms} # 将 platforms 列表中的每个字典转换为一个以 platform_name 为键的字典(值还是完整的m)。platform_name 是每个模型平台配置字典中的一个键，用于唯一标识一个模型平台。



@cached(max_size=10, ttl=60, algorithm=CachingAlgorithmFlag.LRU) # 使用缓存来避免在短时间内发送过多请求，以及避免对每个模型多次请求同一平台。缓存会在一分钟之后失效。
def detect_xf_models(xf_url: str) -> Dict[str, List[str]]: # 定义了 detect_xf_models 函数，它接受一个字符串参数 xf_url（Xinference 服务的 URL），并返回一个字典，字典的键是模型类型，值是该类型模型的列表。
    '''
    use cache for xinference model detecting to avoid:
    - too many requests in short intervals
    - multiple requests to one platform for every model
    the cache will be invalidated after one minute
    '''
    xf_model_type_maps = {
        "llm_models": lambda xf_models: [k for k, v in xf_models.items()
                                        if "LLM" == v["model_type"]
                                        and "vision" not in v["model_ability"]],
        "embed_models": lambda xf_models: [k for k, v in xf_models.items()
                                        if "embedding" == v["model_type"]],
        "text2image_models": lambda xf_models: [k for k, v in xf_models.items()
                                                if "image" == v["model_type"]],
        "image2image_models": lambda xf_models: [k for k, v in xf_models.items()
                                                if "image" == v["model_type"]],
        "image2text_models": lambda xf_models: [k for k, v in xf_models.items()
                                                if "LLM" == v["model_type"]
                                                and "vision" in v["model_ability"]],
        "rerank_models": lambda xf_models: [k for k, v in xf_models.items()
                                            if "rerank" == v["model_type"]],
        "speech2text_models": lambda xf_models: [k for k, v in xf_models.items()
                                                if v.get(list(XF_MODELS_TYPES["speech2text"].keys())[0])
                                                in XF_MODELS_TYPES["speech2text"].values()],
        "text2speech_models": lambda xf_models: [k for k, v in xf_models.items()
                                                if v.get(list(XF_MODELS_TYPES["text2speech"].keys())[0])
                                                in XF_MODELS_TYPES["text2speech"].values()],
    }
    models = {}
    try:
        from xinference_client import RESTfulClient as Client # 这段代码尝试从 xinference_client 导入 RESTfulClient 类
        xf_client = Client(xf_url)# 使用Client来连接到 Xinference 服务，获取模型列表。
        xf_models = xf_client.list_models()
        for m_type, filter in xf_model_type_maps.items(): # 使用 xf_model_type_maps 中的 lambda 函数来筛选和分类模型
            models[m_type] = filter(xf_models) # 对于每个模型类型，使用对应的筛选函数 filter 来处理 xf_models 列表，并将筛选结果存储在 models 字典中，以模型类型为键。
    except ImportError: # ImportError：如果 xinference-client 没有安装。
        logger.warning('auto_detect_model needs xinference-client installed. '
                        'Please try "pip install xinference-client". ')
    except requests.exceptions.ConnectionError: # 如果无法连接到 Xinference 服务。
        logger.warning(f"cannot connect to xinference host: {xf_url}, please check your configuration.")
    except Exception as e: # 如果发生其他错误。
        logger.warning(f"error when connect to xinference server({xf_url}): {e}")
    return models # 在成功获取并分类模型后，返回一个包含不同类型模型的字典


'''用于获取配置中的模型列表，并根据提供的参数筛选出符合条件的模型配置。'''
def get_config_models(
        model_name: str = None,
        model_type: Optional[Literal[
            "llm", "embed", "text2image", "image2image", "image2text", "rerank", "speech2text", "text2speech"
        ]] = None,
        platform_name: str = None,
) -> Dict[str, Dict]: # 函数的返回值是一个字典，其中每个键是模型名称，每个值是一个包含模型详细信息的字典
    """
    获取配置的模型列表，返回值为:
    {model_name: {
        "platform_name": xx,
        "platform_type": xx,
        "model_type": xx,
        "model_name": xx,
        "api_base_url": xx,
        "api_key": xx,
        "api_proxy": xx,
    }}
    """
    result = {} # 初始化一个空字典，用于存储筛选后的模型配置。
    if model_type is None: # 如果未提供 model_type 参数，则设置一个包含所有模型类型名称的列表 model_types。
        model_types = [
            "llm_models",
            "embed_models",
            "text2image_models",
            "image2image_models",
            "image2text_models",
            "rerank_models",
            "speech2text_models",
            "text2speech_models",
        ]
    else:
        model_types = [f"{model_type}_models"] # 如果提供了 model_type 参数，则构造一个只包含该模型类型名称的列表。

    for m in list(get_config_platforms().values()): # 遍历 get_config_platforms 函数返回的平台配置字典的值。
        if platform_name is not None and platform_name != m.get("platform_name"): # 如果提供了 platform_name 参数且与当前平台的名称不匹配，则跳过当前平台。
            continue

        if m.get("auto_detect_model"): # 如果平台配置了自动检测模型，则尝试从 xinference 服务中获取模型列表。
            if not m.get("platform_type") == "xinference":  # TODO：当前仅支持 xf 自动检测模型
                logger.warning(f"auto_detect_model not supported for {m.get('platform_type')} yet")
                continue
            xf_url = get_base_url(m.get("api_base_url"))
            xf_models = detect_xf_models(xf_url)
            for m_type in model_types:
                # if m.get(m_type) != "auto":
                #     continue
                m[m_type] = xf_models.get(m_type, []) # 将检测到的模型列表（或空列表，如果没有检测到模型）赋值给当前平台配置中对应模型类型的键。

        for m_type in model_types:
            models = m.get(m_type, []) # 从当前平台配置 m 中获取对应模型类型 m_type 的模型列表。如果没有找到对应的模型类型，则默认为空列表。
            if models == "auto": # 如果模型列表设置为 “auto”，则记录警告信息。
                logger.warning("you should not set `auto` without auto_detect_model=True")
                continue
            elif not models: # 如果没有配置模型，则跳过当前模型类型。
                continue
            for m_name in models: # 遍历模型列表。
                if model_name is None or model_name == m_name: # 如果未提供 model_name 参数或提供的模型名称与当前模型名称匹配，则将模型配置添加到 result 字典中。
                    result[m_name] = {
                        "platform_name": m.get("platform_name"),
                        "platform_type": m.get("platform_type"),
                        "model_type": m_type.split("_")[0],
                        "model_name": m_name,
                        "api_base_url": m.get("api_base_url"),
                        "api_key": m.get("api_key"),
                        "api_proxy": m.get("api_proxy"),
                    }
    return result


def get_model_info(
        model_name: str = None, platform_name: str = None, multiple: bool = False
) -> Dict:
    """
    获取配置的模型信息，主要是 api_base_url, api_key
    如果指定 multiple=True，则返回所有重名模型；否则仅返回第一个
    """
    result = get_config_models(model_name=model_name, platform_name=platform_name)
    if len(result) > 0:
        if multiple:
            return result
        else:
            return list(result.values())[0]
    else:
        return {}

'''获取默认的大语言模型'''
def get_default_llm():
    available_llms = list(get_config_models(model_type="llm").keys()) # 调用函数，传入参数 model_type="llm" 来获取所有配置的语言模型，并将它们的键（即模型名称）转换成列表
    if Settings.model_settings.DEFAULT_LLM_MODEL in available_llms: # 检查默认的语言模型（通过 Settings.model_settings.DEFAULT_LLM_MODEL 获取）是否在可用的语言模型列表中
        return Settings.model_settings.DEFAULT_LLM_MODEL # 如果默认语言模型在可用列表中，则返回该默认模型。
    else:
        logger.warning(f"default llm model {Settings.model_settings.DEFAULT_LLM_MODEL} is not found in available llms, "
                       f"using {available_llms[0]} instead") # 如果默认语言模型不在可用列表中，记录一条警告信息，并使用列表中的第一个模型作为默认模型。
        return available_llms[0] # 返回列表中的第一个语言模型作为默认模型。

'''获取默认嵌入模型'''
def get_default_embedding():
    available_embeddings = list(get_config_models(model_type="embed").keys()) # 调用函数，传入参数 model_type="embed" 来获取所有配置的嵌入模型，并将它们的键（即模型名称）转换成列表。
    if Settings.model_settings.DEFAULT_EMBEDDING_MODEL in available_embeddings:
        return Settings.model_settings.DEFAULT_EMBEDDING_MODEL
    else:
        logger.warning(f"default embedding model {Settings.model_settings.DEFAULT_EMBEDDING_MODEL} is not found in "
                       f"available embeddings, using {available_embeddings[0]} instead")
        return available_embeddings[0]

'''
get_ChatOpenAI 函数用于创建和配置一个 ChatOpenAI 实例，这是一个封装了 OpenAI 聊天模型（如 ChatGPT）的类。
该函数接受多个参数来定制化模型的行为，并处理一些额外的逻辑，比如本地包装 API 的使用、参数清理等。
'''
def get_ChatOpenAI(
        model_name: str = get_default_llm(),
        temperature: float = Settings.model_settings.TEMPERATURE,
        max_tokens: int = Settings.model_settings.MAX_TOKENS,
        streaming: bool = True,
        callbacks: List[Callable] = [],
        verbose: bool = True,
        local_wrap: bool = False,  # use local wrapped api
        **kwargs: Any,
) -> ChatOpenAI:
    """
        获取配置的模型列表，返回值为:
        {model_name: {
            "platform_name": xx,
            "platform_type": xx,
            "model_type": xx,
            "model_name": xx,
            "api_base_url": xx,
            "api_key": xx,
            "api_proxy": xx,
        }}
        """
    model_info = get_model_info(model_name)
    params = dict(
        streaming=streaming,
        verbose=verbose,
        callbacks=callbacks,
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        **kwargs,
    )
    # remove paramters with None value to avoid openai validation error
    for k in list(params):
        if params[k] is None:
            params.pop(k)

    try:
        if local_wrap: # 根据 local_wrap 参数决定是使用本地包装的 API 还是远程 AP
            params.update(
                openai_api_base=f"{api_address()}/v1",
                openai_api_key="EMPTY",
            )
        else:
            params.update(
                openai_api_base=model_info.get("api_base_url"),
                openai_api_key=model_info.get("api_key"),
                openai_proxy=model_info.get("api_proxy"),
            )
        model = ChatOpenAI(**params)
    except Exception as e:
        logger.exception(f"failed to create ChatOpenAI for model: {model_name}.")
        model = None
    return model


def get_OpenAI(
        model_name: str,
        temperature: float,
        max_tokens: int = Settings.model_settings.MAX_TOKENS,
        streaming: bool = True,
        echo: bool = True,
        callbacks: List[Callable] = [],
        verbose: bool = True,
        local_wrap: bool = False,  # use local wrapped api
        **kwargs: Any,
) -> OpenAI:
    # TODO: 从API获取模型信息
    model_info = get_model_info(model_name)
    params = dict(
        streaming=streaming,
        verbose=verbose,
        callbacks=callbacks,
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        echo=echo,
        **kwargs,
    )
    try:
        if local_wrap:
            params.update(
                openai_api_base=f"{api_address()}/v1",
                openai_api_key="EMPTY",
            )
        else:
            params.update(
                openai_api_base=model_info.get("api_base_url"),
                openai_api_key=model_info.get("api_key"),
                openai_proxy=model_info.get("api_proxy"),
            )
        model = OpenAI(**params)
    except Exception as e:
        logger.exception(f"failed to create OpenAI for model: {model_name}.")
        model = None
    return model


def get_Embeddings(
    embed_model: str = None,
    local_wrap: bool = False,  # use local wrapped api
) -> Embeddings:
    from langchain_community.embeddings import OllamaEmbeddings
    from langchain_openai import OpenAIEmbeddings

    from chatchat.server.localai_embeddings import (
        LocalAIEmbeddings,
    )

    embed_model = embed_model or get_default_embedding()
    model_info = get_model_info(model_name=embed_model)
    params = dict(model=embed_model)
    try:
        if local_wrap:
            params.update(
                openai_api_base=f"{api_address()}/v1",
                openai_api_key="EMPTY",
            )
        else:
            params.update(
                openai_api_base=model_info.get("api_base_url"),
                openai_api_key=model_info.get("api_key"),
                openai_proxy=model_info.get("api_proxy"),
            )
        if model_info.get("platform_type") == "openai":
            return OpenAIEmbeddings(**params)
        elif model_info.get("platform_type") == "ollama":
            return OllamaEmbeddings(
                base_url=model_info.get("api_base_url").replace("/v1", ""),
                model=embed_model,
            )
        else:
            return LocalAIEmbeddings(**params)
    except Exception as e:
        logger.exception(f"failed to create Embeddings for model: {embed_model}.")


def check_embed_model(embed_model: str = None) -> Tuple[bool, str]:
    '''
    check weather embed_model accessable, use default embed model if None
    '''
    embed_model = embed_model or get_default_embedding()
    embeddings = get_Embeddings(embed_model=embed_model)
    try:
        embeddings.embed_query("this is a test")
        return True, ""
    except Exception as e:
        msg = f"failed to access embed model '{embed_model}': {e}"
        logger.error(msg)
        return False, msg

''''''
def get_OpenAIClient(
        platform_name: str = None,
        model_name: str = None,
        is_async: bool = True,
) -> Union[openai.Client, openai.AsyncClient]: # 表示函数返回一个同步或异步的 OpenAI 客户端实例。
    """
    construct an openai Client for specified platform or model
    """
    if platform_name is None:
        platform_info = get_model_info( # 根据模型名称获取平台（XF）信息
            model_name=model_name, platform_name=platform_name
        )
        if platform_info is None:
            raise RuntimeError(
                f"cannot find configured platform for model: {model_name}"
            )
        platform_name = platform_info.get("platform_name")
    platform_info = get_config_platforms().get(platform_name) # 从本项目支持的所有平台中选出platform_name（XF）则个平台的信息
    assert platform_info, f"cannot find configured platform: {platform_name}" # 这是一个断言语句，用于检查platform_info是否为真
    params = { # 构造客户端基础参数params,包括 base_url 和 api_key。
        "base_url": platform_info.get("api_base_url"),
        "api_key": platform_info.get("api_key"),
    }
    httpx_params = {}
    if api_proxy := platform_info.get("api_proxy"): # 如果平台配置中有代理设置 (api_proxy)，则额外构造 httpx_params，包括代理地址和其他 HTTPX 客户端传输选项。
        httpx_params = { # todo 这里的配置还不懂
            "proxies": api_proxy,
            "transport": httpx.HTTPTransport(local_address="0.0.0.0"),
        }

    if is_async: # 根据 is_async 参数决定创建同步还是异步客户端
        if httpx_params: # 如果有 httpx_params，则使用这些参数创建 httpx.AsyncClient 并将其作为 http_client 参数传递给 openai.AsyncClient。
            params["http_client"] = httpx.AsyncClient(**httpx_params)
        return openai.AsyncClient(**params)
    else:
        if httpx_params:
            params["http_client"] = httpx.Client(**httpx_params)
        return openai.Client(**params)


class MsgType:
    TEXT = 1
    IMAGE = 2
    AUDIO = 3
    VIDEO = 4


class BaseResponse(BaseModel):
    code: int = Field(200, description="API status code")
    msg: str = Field("success", description="API status message")
    data: Any = Field(None, description="API data")

    class Config:
        json_schema_extra = {
            "example": {
                "code": 200,
                "msg": "success",
            }
        }


class ListResponse(BaseResponse):
    data: List[Any] = Field(..., description="List of data")

    class Config:
        json_schema_extra = {
            "example": {
                "code": 200,
                "msg": "success",
                "data": ["doc1.docx", "doc2.pdf", "doc3.txt"],
            }
        }


class ChatMessage(BaseModel):
    question: str = Field(..., description="Question text")
    response: str = Field(..., description="Response text")
    history: List[List[str]] = Field(..., description="History text")
    source_documents: List[str] = Field(
        ..., description="List of source documents and their scores"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "question": "工伤保险如何办理？",
                "response": "根据已知信息，可以总结如下：\n\n1. 参保单位为员工缴纳工伤保险费，以保障员工在发生工伤时能够获得相应的待遇。\n"
                            "2. 不同地区的工伤保险缴费规定可能有所不同，需要向当地社保部门咨询以了解具体的缴费标准和规定。\n"
                            "3. 工伤从业人员及其近亲属需要申请工伤认定，确认享受的待遇资格，并按时缴纳工伤保险费。\n"
                            "4. 工伤保险待遇包括工伤医疗、康复、辅助器具配置费用、伤残待遇、工亡待遇、一次性工亡补助金等。\n"
                            "5. 工伤保险待遇领取资格认证包括长期待遇领取人员认证和一次性待遇领取人员认证。\n"
                            "6. 工伤保险基金支付的待遇项目包括工伤医疗待遇、康复待遇、辅助器具配置费用、一次性工亡补助金、丧葬补助金等。",
                "history": [
                    [
                        "工伤保险是什么？",
                        "工伤保险是指用人单位按照国家规定，为本单位的职工和用人单位的其他人员，缴纳工伤保险费，"
                        "由保险机构按照国家规定的标准，给予工伤保险待遇的社会保险制度。",
                    ]
                ],
                "source_documents": [
                    "出处 [1] 广州市单位从业的特定人员参加工伤保险办事指引.docx：\n\n\t"
                    "( 一)  从业单位  (组织)  按“自愿参保”原则，  为未建 立劳动关系的特定从业人员单项参加工伤保险 、缴纳工伤保 险费。",
                    "出处 [2] ...",
                    "出处 [3] ...",
                ],
            }
        }


def run_async(cor):
    """
    在同步环境中运行异步代码.
    """
    try:
        loop = asyncio.get_event_loop()
    except:
        loop = asyncio.new_event_loop()
    return loop.run_until_complete(cor)


def iter_over_async(ait, loop=None):
    """
    将异步生成器封装成同步生成器.
    """
    ait = ait.__aiter__()

    async def get_next():
        try:
            obj = await ait.__anext__()
            return False, obj
        except StopAsyncIteration:
            return True, None

    if loop is None:
        try:
            loop = asyncio.get_event_loop()
        except:
            loop = asyncio.new_event_loop()

    while True:
        done, obj = loop.run_until_complete(get_next())
        if done:
            break
        yield obj

'''函数的作用是修改一个 FastAPI 应用，使得其文档页面（Swagger UI 和 ReDoc）不依赖于外部的 CDN（内容分发网络），而是使用本地静态文件(相当于本地部署swagger?)。'''
def MakeFastAPIOffline(
        app: FastAPI,
        static_dir=Path(__file__).parent / "api_server" / "static",
        static_url="/static-offline-docs",# 指定的静态文件目录（默认为 api_server/static）挂载到应用中，通过特定 URL 路径（如 /static-offline-docs）提供服务
        docs_url: Optional[str] = "/docs", # 定义新的 /docs 和 /redoc 路由来返回自定义的 Swagger UI 和 ReDoc HTML 页面,这些页面会指向本地静态资源（例如 JavaScript 文件、CSS 文件等），而不是 CDN 上的资源
        redoc_url: Optional[str] = "/redoc",
) -> None:
    """patch the FastAPI obj that doesn't rely on CDN for the documentation page"""
    from fastapi import Request
    from fastapi.openapi.docs import (
        get_redoc_html,
        get_swagger_ui_html,
        get_swagger_ui_oauth2_redirect_html,
    )
    from fastapi.staticfiles import StaticFiles
    from starlette.responses import HTMLResponse

    openapi_url = app.openapi_url
    swagger_ui_oauth2_redirect_url = app.swagger_ui_oauth2_redirect_url

    def remove_route(url: str) -> None:
        """
        为了替换默认的文档页面，首先需要删除 FastAPI 自动添加的相关路由。
        """
        index = None
        for i, r in enumerate(app.routes):
            if r.path.lower() == url.lower():
                index = i
                break
        if isinstance(index, int):
            app.routes.pop(index)

    '''将指定的静态文件目录（默认为 api_server/static）挂载到应用中，通过特定 URL 路径（如 /static-offline-docs）提供服务。'''
    app.mount(
        static_url,
        StaticFiles(directory=Path(static_dir).as_posix()),
        name="static-offline-docs",
    )

    '''定义新的 /docs 路由来返回自定义的 Swagger UI 页面。这些页面会指向本地静态资源（例如 JavaScript 文件、CSS 文件等），而不是 CDN 上的资源。'''
    if docs_url is not None:
        remove_route(docs_url) # 删除原有的路由
        remove_route(swagger_ui_oauth2_redirect_url)

        # Define the doc and redoc pages, pointing at the right files
        @app.get(docs_url, include_in_schema=False)
        async def custom_swagger_ui_html(request: Request) -> HTMLResponse:
            root = request.scope.get("root_path")
            favicon = f"{root}{static_url}/favicon.png"
            return get_swagger_ui_html(
                openapi_url=f"{root}{openapi_url}",
                title=app.title + " - Swagger UI",
                oauth2_redirect_url=swagger_ui_oauth2_redirect_url,
                swagger_js_url=f"{root}{static_url}/swagger-ui-bundle.js",
                swagger_css_url=f"{root}{static_url}/swagger-ui.css",
                swagger_favicon_url=favicon,
            )

        '''如果启用了 OAuth2 支持，则还会为 OAuth2 重定向提供一个单独的路由。'''
        @app.get(swagger_ui_oauth2_redirect_url, include_in_schema=False)
        async def swagger_ui_redirect() -> HTMLResponse:
            return get_swagger_ui_oauth2_redirect_html()

    '''定义新的 /redoc 路由来返回自定义的 ReDoc HTML 页面。这些页面会指向本地静态资源（例如 JavaScript 文件、CSS 文件等），而不是 CDN 上的资源。'''
    if redoc_url is not None:
        remove_route(redoc_url)

        @app.get(redoc_url, include_in_schema=False)
        async def redoc_html(request: Request) -> HTMLResponse:
            root = request.scope.get("root_path")
            favicon = f"{root}{static_url}/favicon.png"

            return get_redoc_html(
                openapi_url=f"{root}{openapi_url}",
                title=app.title + " - ReDoc",
                redoc_js_url=f"{root}{static_url}/redoc.standalone.js",
                with_google_fonts=False,
                redoc_favicon_url=favicon,
            )


# 从model_config中获取模型信息
# TODO: 移出模型加载后，这些功能需要删除或改变实现

# def list_embed_models() -> List[str]:
#     '''
#     get names of configured embedding models
#     '''
#     return list(MODEL_PATH["embed_model"])


# def get_model_path(model_name: str, type: str = None) -> Optional[str]:
#     if type in MODEL_PATH:
#         paths = MODEL_PATH[type]
#     else:
#         paths = {}
#         for v in MODEL_PATH.values():
#             paths.update(v)

#     if path_str := paths.get(model_name):  # 以 "chatglm-6b": "THUDM/chatglm-6b-new" 为例，以下都是支持的路径
#         path = Path(path_str)
#         if path.is_dir():  # 任意绝对路径
#             return str(path)

#         root_path = Path(MODEL_ROOT_PATH)
#         if root_path.is_dir():
#             path = root_path / model_name
#             if path.is_dir():  # use key, {MODEL_ROOT_PATH}/chatglm-6b
#                 return str(path)
#             path = root_path / path_str
#             if path.is_dir():  # use value, {MODEL_ROOT_PATH}/THUDM/chatglm-6b-new
#                 return str(path)
#             path = root_path / path_str.split("/")[-1]
#             if path.is_dir():  # use value split by "/", {MODEL_ROOT_PATH}/chatglm-6b-new
#                 return str(path)
#         return path_str  # THUDM/chatglm06b


def api_address(is_public: bool = False) -> str:
    '''
    允许用户在 basic_settings.API_SERVER 中配置 public_host, public_port
    以便使用云服务器或反向代理时生成正确的公网 API 地址（如知识库文档下载链接）
    '''
    from chatchat.settings import Settings

    server = Settings.basic_settings.API_SERVER
    if is_public:
        host = server.get("public_host", "127.0.0.1") # todo 原
        port = server.get("public_port", "7861")
    else:
        host = server.get("host", "127.0.0.1")
        port = server.get("port", "7861")
        if host == "0.0.0.0":
            host = "127.0.0.1"
    return f"http://{host}:{port}"


def webui_address() -> str:
    from chatchat.settings import Settings

    host = Settings.basic_settings.WEBUI_SERVER["host"]
    port = Settings.basic_settings.WEBUI_SERVER["port"]
    return f"http://{host}:{port}"


def get_prompt_template(type: str, name: str) -> Optional[str]:
    """
    从prompt_config中加载模板内容
    type: 对应于 model_settings.llm_model_config 模型类别其中的一种，以及 "rag"，如果有新功能，应该进行加入。
    """

    from chatchat.settings import Settings

    return Settings.prompt_settings.model_dump().get(type, {}).get(name)


def set_httpx_config(
        timeout: float = Settings.basic_settings.HTTPX_DEFAULT_TIMEOUT,
        proxy: Union[str, Dict] = None,
        unused_proxies: List[str] = [],
):
    """
    设置httpx默认timeout。httpx默认timeout是5秒，在请求LLM回答时不够用。
    将本项目相关服务加入无代理列表，避免fastchat的服务器请求错误。(windows下无效)
    对于chatgpt等在线API，如要使用代理需要手动配置。搜索引擎的代理如何处置还需考虑。
    """

    import os

    import httpx

    httpx._config.DEFAULT_TIMEOUT_CONFIG.connect = timeout
    httpx._config.DEFAULT_TIMEOUT_CONFIG.read = timeout
    httpx._config.DEFAULT_TIMEOUT_CONFIG.write = timeout

    # 在进程范围内设置系统级代理
    proxies = {}
    if isinstance(proxy, str):
        for n in ["http", "https", "all"]:
            proxies[n + "_proxy"] = proxy
    elif isinstance(proxy, dict):
        for n in ["http", "https", "all"]:
            if p := proxy.get(n):
                proxies[n + "_proxy"] = p
            elif p := proxy.get(n + "_proxy"):
                proxies[n + "_proxy"] = p

    for k, v in proxies.items():
        os.environ[k] = v

    # set host to bypass proxy
    no_proxy = [
        x.strip() for x in os.environ.get("no_proxy", "").split(",") if x.strip()
    ]
    no_proxy += [
        # do not use proxy for locahost
        "http://127.0.0.1",
        "http://localhost",
    ]
    # do not use proxy for user deployed fastchat servers
    for x in unused_proxies:
        host = ":".join(x.split(":")[:2])
        if host not in no_proxy:
            no_proxy.append(host)
    os.environ["NO_PROXY"] = ",".join(no_proxy)

    def _get_proxies():
        return proxies

    import urllib.request

    urllib.request.getproxies = _get_proxies


def run_in_thread_pool(
        func: Callable,
        params: List[Dict] = [],
) -> Generator:
    """
    在线程池中批量运行任务，并将运行结果以生成器的形式返回。
    请确保任务中的所有操作是线程安全的，任务函数请全部使用关键字参数。
    """
    tasks = []
    with ThreadPoolExecutor() as pool:
        for kwargs in params:
            tasks.append(pool.submit(func, **kwargs))

        for obj in as_completed(tasks):
            try:
                yield obj.result()
            except Exception as e:
                logger.exception(f"error in sub thread: {e}")


def run_in_process_pool(
        func: Callable,
        params: List[Dict] = [],
) -> Generator:
    """
    在线程池中批量运行任务，并将运行结果以生成器的形式返回。
    请确保任务中的所有操作是线程安全的，任务函数请全部使用关键字参数。
    """
    tasks = []
    max_workers = None
    if sys.platform.startswith("win"):
        max_workers = min(
            mp.cpu_count(), 60
        )  # max_workers should not exceed 60 on windows
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        for kwargs in params:
            tasks.append(pool.submit(func, **kwargs))

        for obj in as_completed(tasks):
            try:
                yield obj.result()
            except Exception as e:
                logger.exception(f"error in sub process: {e}")


def get_httpx_client(
        use_async: bool = False,
        proxies: Union[str, Dict] = None,
        timeout: float = Settings.basic_settings.HTTPX_DEFAULT_TIMEOUT,
        unused_proxies: List[str] = [],
        **kwargs,
) -> Union[httpx.Client, httpx.AsyncClient]:
    """
    helper to get httpx client with default proxies that bypass local addesses.
    """
    default_proxies = {
        # do not use proxy for locahost
        "all://127.0.0.1": None,
        "all://localhost": None,
    }
    # do not use proxy for user deployed fastchat servers
    for x in unused_proxies:
        host = ":".join(x.split(":")[:2])
        default_proxies.update({host: None})

    # get proxies from system envionrent
    # proxy not str empty string, None, False, 0, [] or {}
    default_proxies.update(
        {
            "http://": (
                os.environ.get("http_proxy")
                if os.environ.get("http_proxy")
                   and len(os.environ.get("http_proxy").strip())
                else None
            ),
            "https://": (
                os.environ.get("https_proxy")
                if os.environ.get("https_proxy")
                   and len(os.environ.get("https_proxy").strip())
                else None
            ),
            "all://": (
                os.environ.get("all_proxy")
                if os.environ.get("all_proxy")
                   and len(os.environ.get("all_proxy").strip())
                else None
            ),
        }
    )
    for host in os.environ.get("no_proxy", "").split(","):
        if host := host.strip():
            # default_proxies.update({host: None}) # Origin code
            default_proxies.update(
                {"all://" + host: None}
            )  # PR 1838 fix, if not add 'all://', httpx will raise error

    # merge default proxies with user provided proxies
    if isinstance(proxies, str):
        proxies = {"all://": proxies}

    if isinstance(proxies, dict):
        default_proxies.update(proxies)

    # construct Client
    kwargs.update(timeout=timeout, proxies=default_proxies)

    if use_async:
        return httpx.AsyncClient(**kwargs)
    else:
        return httpx.Client(**kwargs)


def get_server_configs() -> Dict:
    """
    获取configs中的原始配置项，供前端使用
    """
    _custom = {
        "api_address": api_address(),
    }

    return {**{k: v for k, v in locals().items() if k[0] != "_"}, **_custom}


def get_temp_dir(id: str = None) -> Tuple[str, str]:
    """
    创建一个临时目录，返回（路径，文件夹名称）
    """
    import uuid

    from chatchat.settings import Settings

    if id is not None:  # 如果指定的临时目录已存在，直接返回
        path = os.path.join(Settings.basic_settings.BASE_TEMP_DIR, id)
        if os.path.isdir(path):
            return path, id

    id = uuid.uuid4().hex
    path = os.path.join(Settings.basic_settings.BASE_TEMP_DIR, id)
    os.mkdir(path)
    return path, id


'''更新一个名为 search_local_knowledgebase 的工具。该工具用于搜索本地知识库，并且这个函数确保工具的描述和参数选项是最新的，反映了数据库中当前可用的知识库信息。'''
def update_search_local_knowledgebase_tool():
    import re # 正则表达式模块，用于字符串处理。

    from chatchat.server.agent.tools_factory import tools_registry # 工具注册管理包
    from chatchat.server.db.repository.knowledge_base_repository import list_kbs_from_db # 列出数据库中的所有知识库包

    kbs = list_kbs_from_db() # 列出数据库中的所有知识库
    # 此为格式化字符串模板，用于构建工具的描述。它包含两个占位符 {KB_info} 和 {key}，稍后将被替换为实际值。
    template = "Use local knowledgebase from one or more of these:\n{KB_info}\n to get information，Only local data on this knowledge use this tool. The 'database' should be one of the above [{key}]."
    KB_info_str = "\n".join([f"{kb.kb_name}: {kb.kb_info}" for kb in kbs]) # 通过遍历 kbs 列表，构建一个包含每个知识库名称及其信息的字符串，每条记录之间用换行符分隔。
    KB_name_info_str = "\n".join([f"{kb.kb_name}" for kb in kbs]) # 遍历 kbs 列表，但只构建包含知识库名称的字符串，用于限制工具参数中的选择项。
    template_knowledge = template.format(KB_info=KB_info_str, key=KB_name_info_str) # 使用 format 方法填充模板字符串中的占位符，生成最终的工具描述文本。

    search_local_knowledgebase_tool = tools_registry._TOOLS_REGISTRY.get( # 从 _TOOLS_REGISTRY 字典中获取key为 "search_local_knowledgebase"(就是本地知识库) 的工具实例。
        "search_local_knowledgebase" # todo 这里不明白的话在swagger文档里请求一次/tools接口就明白了。
    )
    if search_local_knowledgebase_tool:
        search_local_knowledgebase_tool.description = " ".join(
            re.split(r"\n+\s*", template_knowledge)
        )
        search_local_knowledgebase_tool.args["database"]["choices"] = [ # 设置工具参数 args["database"]["choices"] 为当前所有知识库的名称列表，确保用户只能选择存在的知识库。
            kb.kb_name for kb in kbs
        ]


'''
根据提供的工具名称 name 来获取对应的工具实例或者返回所有注册工具的字典。用于检索和管理聊天机器人框架中工具的函数。
它能够根据需要提供单个工具的访问或者提供多个工具的访问。这样的设计使得框架能够灵活地管理和使用不同的工具。
'''
def get_tool(name: str = None) -> Union[BaseTool, Dict[str, BaseTool]]:
    import importlib # 导入 Python 模块的新方法和重新加载模块的功能。

    from chatchat.server.agent import tools_factory

    importlib.reload(tools_factory) # 重新加载了 tools_factory 模块。这样做通常是为了确保模块中的最新代码被使用，尤其是在开发过程中模块内容可能发生变化。

    from chatchat.server.agent.tools_factory import tools_registry

    update_search_local_knowledgebase_tool() # 动态知识库更新
    if name is None:
        return tools_registry._TOOLS_REGISTRY
    else:
        return tools_registry._TOOLS_REGISTRY.get(name)


def get_tool_config(name: str = None) -> Dict:
    from chatchat.settings import Settings

    if name is None:
        return Settings.tool_settings.model_dump()
    else:
        return Settings.tool_settings.model_dump().get(name, {})


def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        return sock.connect_ex(("localhost", port)) == 0


if __name__ == "__main__":
    # for debug
    print(get_default_llm()) # qwen2-instruct
    print(get_default_embedding()) # bge-m3
    platforms = get_config_platforms()
    print(platforms)
    models = get_config_models()
    print(models)
    model_info = get_model_info(platform_name="xinference-auto")
    print(model_info)
    print(1)
