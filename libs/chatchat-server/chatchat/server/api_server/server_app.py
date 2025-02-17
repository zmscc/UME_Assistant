'''
这个脚本是一个命令行工具，用于启动一个 FastAPI 应用，该应用提供了一系列的 API 端点，包括聊天、知识库、工具、OpenAI 接口和服务器状态相关的路由。
脚本还处理静态文件服务和可能的 SSL 配置。
'''
import argparse
import os
from typing import Literal

import uvicorn
from fastapi import Body, FastAPI
from fastapi.middleware.cors import CORSMiddleware # 添加 CORS 中间件，使得 API 服务器能够接受来自不同域的请求
from fastapi.staticfiles import StaticFiles # 挂载静态文件，可以让用户可以直接访问存储在服务器上的图片和其他资源
from starlette.responses import RedirectResponse

from chatchat import __version__
from chatchat.settings import Settings
from chatchat.server.api_server.chat_routes import chat_router
from chatchat.server.api_server.kb_routes import kb_router
from chatchat.server.api_server.openai_routes import openai_router
from chatchat.server.api_server.server_routes import server_router
from chatchat.server.api_server.tool_routes import tool_router
from chatchat.server.chat.completion import completion
from chatchat.server.utils import MakeFastAPIOffline


def create_app(run_mode: str = None):
    app = FastAPI(title="航旅助手 API Server", version=__version__)
    MakeFastAPIOffline(app)
    # Add CORS middleware to allow all origins
    # 在config.py中设置OPEN_DOMAIN=True，允许跨域
    # set OPEN_DOMAIN=True in config.py to allow cross-domain
    '''CORS 中间件的作用就是在服务器端添加 HTTP 响应头，告诉浏览器哪些来源（域名、协议或端口）可以访问服务器上的资源。
       通过在 FastAPI 应用中添加 CORS 中间件，你可以配置你的 API 服务器允许来自特定域或者所有域的请求，从而解决跨域问题。
       在下面代码中，如果设置了 OPEN_CROSS_DOMAIN 为真，则会添加 CORS 中间件，并允许所有来源的请求'''
    if Settings.basic_settings.OPEN_CROSS_DOMAIN:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"], # 允许所有来源
            allow_credentials=True,
            allow_methods=["*"], # 允许所有HTTP方法
            allow_headers=["*"], # 允许所有头部信息
        )

    @app.get("/", summary="swagger 文档", include_in_schema=False) # document 函数并不是单独定义的，而是通过装饰器 @app.get("/") 与 FastAPI 应用实例 app 关联起来的。这意味着 document 函数是 FastAPI 应用的一部分，而不是独立存在。当用户访问根路径 (/) 时，FastAPI 会调用 document 函数来处理请求，并执行重定向到 /docs。
    async def document():
        return RedirectResponse(url="/docs") # 返回一个 HTTP 重定向响应，告诉客户端浏览器去请求 /docs 路径，这是默认的 Swagger UI 页面地址。

    # 将多个不同的路由器（chat_router, kb_router, tool_router, openai_router, server_router）包含到主应用中。每个路由器代表一组相关的 API 端点
    app.include_router(chat_router) # 这行代码的作用是将 chat_router 中定义的所有路由添加到 app 中，使得这些路由可以开始接收和处理 HTTP 请求。
    app.include_router(kb_router)
    app.include_router(tool_router)
    app.include_router(openai_router)
    app.include_router(server_router)

    # 其它接口
    app.post(
        "/other/completion", # 创建一个新的 POST API 端点，路径为 /other/completion。
        tags=["Other"], # 给这个端点添加标签 "Other"，这有助于在文档中对端点进行分类
        summary="要求llm模型补全(通过LLMChain)",
    )(completion) # 客户端应用程序将发送一个 POST 请求到服务器上的 /other/completion 端点，携带了用户的查询("故宫有多大？")和其他配置参数。服务器接收到这个请求后，调用 completion 函数来处理

    '''静态文件指的是那些不需要服务器端处理就可以直接提供给客户端的文件，比如图片、CSS 文件、JavaScript 文件等。
    为了让用户能够直接访问这些资源，你需要将它们挂载到 FastAPI 应用中，这样当用户请求特定路径时，FastAPI 就可以直接返回相应的静态文件。
    在下面的代码中，应用程序挂载了两个静态文件目录：/media 和 /img。
    当有人访问 http://yourserver/media/filename 或 http://yourserver/img/filename 时，FastAPI 会查找指定的文件夹并尝试返回对应的文件给用户。'''
    # 媒体文件
    app.mount("/media", StaticFiles(directory=Settings.basic_settings.MEDIA_PATH), name="media")
    # 项目相关图片
    img_dir = str(Settings.basic_settings.IMG_DIR)
    app.mount("/img", StaticFiles(directory=img_dir), name="img")

    return app

# 2.启动 Uvicorn 服务器
def run_api(host, port, **kwargs):
    if kwargs.get("ssl_keyfile") and kwargs.get("ssl_certfile"):
        uvicorn.run( # 启动 Uvicorn 服务器，并且开始监听指定主机和端口上的 HTTP/HTTPS 请求。这样，FastAPI 应用就准备好接收来自客户端的请求了。
            app,
            host=host,
            port=port,
            ssl_keyfile=kwargs.get("ssl_keyfile"), # 可选地接受 SSL 证书文件和私钥文件来启用 HTTPS，如果没有提供SSL文件，则启动不带 SSL 的普通 HTTP 服务。
            ssl_certfile=kwargs.get("ssl_certfile"),
        )
    else:
        uvicorn.run(app, host=host, port=port) # 启动 Uvicorn 服务器（启动HTTP服务），并且开始监听指定主机和端口上的 HTTP/HTTPS 请求。这样，FastAPI 应用就准备好接收来自客户端的请求了。


app = create_app() # 1.调用 create_app() 函数来创建 FastAPI 应用程序实例，并将其赋值给变量 app。这个实例包含了所有路由、中间件和其他配置，准备被 Uvicorn 服务器托管。


if __name__ == "__main__": # 这个条件确保以下代码仅在脚本直接运行时执行，而不是在作为模块导入到其他脚本中时执行。用于区分脚本是被直接执行还是被导入为库的一部分。
    parser = argparse.ArgumentParser( # 创建一个解析器对象，用于处理命令行参数。
        prog="langchain-ChatGLM",
        description="About langchain-ChatGLM, local knowledge based ChatGLM with langchain"
        " ｜ 基于本地知识库的 ChatGLM 问答",
    )
    parser.add_argument("--host", type=str, default="0.0.0.0") # 添加命令行参数
    parser.add_argument("--port", type=int, default=7861) # 添加命令行参数
    parser.add_argument("--ssl_keyfile", type=str)
    parser.add_argument("--ssl_certfile", type=str)
    # 初始化消息
    args = parser.parse_args() # parse_args()解析命令行参数并返回一个命名空间对象 args，其中包含所有传入的参数值。
    args_dict = vars(args) # 将命名空间对象转换为字典形式，方便后续访问参数值。args_dict 是一个字典，键是参数名，值是对应的参数值。

    run_api( # run_api 函数时，实际上会将之前创建的 app 实例作为参数传递给 Uvicorn 的 run 方法。注：run_api 函数和 app 变量位于同一个模块中，因此 run_api 可以直接访问 app。
        host=args.host,
        port=args.port,
        ssl_keyfile=args.ssl_keyfile,
        ssl_certfile=args.ssl_certfile,
    )
