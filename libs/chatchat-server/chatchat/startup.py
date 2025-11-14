import asyncio
import logging
import logging.config
import multiprocessing as mp
import os
import sys
import traceback
from contextlib import asynccontextmanager
from multiprocessing import Process

# 设置numexpr最大线程数，默认为CPU核心数
try:
    import numexpr

    n_cores = numexpr.utils.detect_number_of_cores()
    os.environ["NUMEXPR_MAX_THREADS"] = str(n_cores)
except:
    pass

import click
from typing import Dict, List

from fastapi import FastAPI

from chatchat.utils import build_logger


logger = build_logger()


def _set_app_event(app: FastAPI, started_event: mp.Event = None):
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if started_event is not None:
            started_event.set() # started_event.set() 的作用是：将事件状态设为“已触发”（set）,其他等待这个事件的进程会从 started_event.wait() 中醒来
        yield

    app.router.lifespan_context = lifespan


def run_api_server(
    started_event: mp.Event = None, run_mode: str = None
):
    import uvicorn
    from chatchat.utils import (
        get_config_dict,
        get_log_file,
        get_timestamp_ms,
    )

    from chatchat.settings import Settings
    from chatchat.server.api_server.server_app import create_app
    from chatchat.server.utils import set_httpx_config

    logger.info(f"Api MODEL_PLATFORMS: {Settings.model_settings.MODEL_PLATFORMS}")
    set_httpx_config()
    app = create_app(run_mode=run_mode)
    _set_app_event(app, started_event) #在 FastAPI 应用启动完成后，自动触发一个“事件”（Event），通知其他进程：“我已经准备好了！”

    host = Settings.basic_settings.API_SERVER["host"]
    port = Settings.basic_settings.API_SERVER["port"]

    logging_conf = get_config_dict(
        "INFO",
        get_log_file(log_path=Settings.basic_settings.LOG_PATH, sub_dir=f"run_api_server_{get_timestamp_ms()}"),
        1024 * 1024 * 1024 * 3,
        1024 * 1024 * 1024 * 3,
    )
    logging.config.dictConfig(logging_conf)  # type: ignore
    uvicorn.run(app, host=host, port=port) # “启动服务器，让 FastAPI 应用对外提供服务”,如果没有这一行，相当于已经有一个功能齐全的手机了，但是没有开机


def run_webui(
    started_event: mp.Event = None, run_mode: str = None
):
    from chatchat.settings import Settings
    from chatchat.server.utils import set_httpx_config
    from chatchat.utils import get_config_dict, get_log_file, get_timestamp_ms

    logger.info(f"Webui MODEL_PLATFORMS: {Settings.model_settings.MODEL_PLATFORMS}")
    set_httpx_config()

    host = Settings.basic_settings.WEBUI_SERVER["host"]
    port = Settings.basic_settings.WEBUI_SERVER["port"]

    script_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "webui.py") # 找到当前目录下的 webui.py 文件路径，相当于告诉 Streamlit：“你要运行这个 Python 脚本作为网页”

    flag_options = { # 用来定制 Streamlit 的各种行为，比如定制主题颜色，页面样式等等。
        "server_address": host,
        "server_port": port,
        "theme_base": "light",
        "theme_primaryColor": "#165dff",
        "theme_secondaryBackgroundColor": "#f5f5f5",
        "theme_textColor": "#000000",
        "global_disableWatchdogWarning": None,
        "global_disableWidgetStateDuplicationWarning": None,
        "global_showWarningOnDirectExecution": None,
        "global_developmentMode": None,
        "global_logLevel": None,
        "global_unitTest": None,
        "global_suppressDeprecationWarnings": None,
        "global_minCachedMessageSize": None,
        "global_maxCachedMessageAge": None,
        "global_storeCachedForwardMessagesInMemory": None,
        "global_dataFrameSerialization": None,
        "logger_level": None,
        "logger_messageFormat": None,
        "logger_enableRich": None,
        "client_caching": None,
        "client_displayEnabled": None,
        "client_showErrorDetails": None,
        "client_toolbarMode": None,
        "client_showSidebarNavigation": None,
        "runner_magicEnabled": None,
        "runner_installTracer": None,
        "runner_fixMatplotlib": None,
        "runner_postScriptGC": None,
        "runner_fastReruns": None,
        "runner_enforceSerializableSessionState": None,
        "runner_enumCoercion": None,
        "server_folderWatchBlacklist": None,
        "server_fileWatcherType": "none",
        "server_headless": None,
        "server_runOnSave": None,
        "server_allowRunOnSave": None,
        "server_scriptHealthCheckEnabled": None,
        "server_baseUrlPath": None,
        "server_enableCORS": None,
        "server_enableXsrfProtection": None,
        "server_maxUploadSize": None,
        "server_maxMessageSize": None,
        "server_enableArrowTruncation": None,
        "server_enableWebsocketCompression": None,
        "server_enableStaticServing": None,
        "browser_serverAddress": None,
        "browser_gatherUsageStats": None,
        "browser_serverPort": None,
        "server_sslCertFile": None,
        "server_sslKeyFile": None,
        "ui_hideTopBar": None,
        "ui_hideSidebarNav": None,
        "magic_displayRootDocString": None,
        "magic_displayLastExprIfNoSemicolon": None,
        "deprecation_showfileUploaderEncoding": None,
        "deprecation_showImageFormat": None,
        "deprecation_showPyplotGlobalUse": None,
        "theme_backgroundColor": None,
        "theme_font": None,
    }

    args = []
    if run_mode == "lite":
        args += [
            "--",
            "lite",
        ]

    try: # 兼容不同版本的 Streamlit
        # for streamlit >= 1.12.1
        from streamlit.web import bootstrap
    except ImportError:
        from streamlit import bootstrap

    logging_conf = get_config_dict(
        "INFO",
        get_log_file(log_path=Settings.basic_settings.LOG_PATH, sub_dir=f"run_webui_{get_timestamp_ms()}"),
        1024 * 1024 * 1024 * 3,
        1024 * 1024 * 1024 * 3,
    )
    logging.config.dictConfig(logging_conf)  # type: ignore
    bootstrap.load_config_options(flag_options=flag_options) # 把 flag_options 加载为全局配置,替代 .streamlit/config.toml
    bootstrap.run(script_dir, False, args, flag_options)# 启动服务器，让 Streamlit 应用对外提供服务，这里最主要就是找到webui.py这个脚本执行。
    started_event.set() # 这行代码相当于子进程对主进程说“老板！我这边服务器已经起来了，可以对外服务了！主进程听到后，就从 .wait() 中醒来，继续执行后面的代码。


def dump_server_info(after_start=False, args=None):
    import platform

    import langchain

    from chatchat import __version__
    from chatchat.settings import Settings
    from chatchat.server.utils import api_address, webui_address

    print("\n")
    print("=" * 30 + "航旅助手 Configuration" + "=" * 30)
    print(f"操作系统：{platform.platform()}.")
    print(f"python版本：{sys.version}")
    print(f"项目版本：{__version__}")
    print(f"langchain版本：{langchain.__version__}")
    print(f"数据目录：{Settings.CHATCHAT_ROOT}")
    print("\n")

    print(f"当前使用的分词器：{Settings.kb_settings.TEXT_SPLITTER_NAME}")

    print(f"默认选用的 Embedding 名称： {Settings.model_settings.DEFAULT_EMBEDDING_MODEL}")

    if after_start:
        print("\n")
        print(f"服务端运行信息：")
        if args.api:
            print(f"    Chatchat Api Server: {api_address()}")
        if args.webui:
            print(f"    Chatchat WEBUI Server: {webui_address()}")
    print("=" * 30 + "航旅助手 Configuration" + "=" * 30)
    print("\n")


async def start_main_server(args):
    import signal
    import time

    from chatchat.utils import (
        get_config_dict,
        get_log_file,
        get_timestamp_ms,
    )

    from chatchat.settings import Settings

    # 使用 get_config_dict 函数配置日志系统，设置日志级别、文件路径和大小限制
    logging_conf = get_config_dict(
        "INFO",
        get_log_file(
            log_path=Settings.basic_settings.LOG_PATH, sub_dir=f"start_main_server_{get_timestamp_ms()}"
        ),
        1024 * 1024 * 1024 * 3,
        1024 * 1024 * 1024 * 3,
    )
    logging.config.dictConfig(logging_conf)  # type: ignore

    def handler(signalname):
        """
        Python 3.9 has `signal.strsignal(signalnum)` so this closure would not be needed.
        Also, 3.8 includes `signal.valid_signals()` that can be used to create a mapping for the same purpose.
        """

        def f(signal_received, frame):
            raise KeyboardInterrupt(f"{signalname} received")

        return f

    # This will be inherited by the child process if it is forked (not spawned)
    signal.signal(signal.SIGINT, handler("SIGINT")) # ctrl+c优雅关闭程序
    signal.signal(signal.SIGTERM, handler("SIGTERM"))

    mp.set_start_method("spawn") # 设置进程启动方法为 “spawn”，这通常用于确保子进程有干净的内存空间。
    manager = mp.Manager() # 创建一个进程管理器，可以用于进程间通信。
    run_mode = None

    if args.all: # 根据 args 参数决定是否启动API服务器和Web UI服务器。
        args.api = True
        args.webui = True

    dump_server_info(args=args) # 打印启动日志信息

    if len(sys.argv) > 1:
        logger.info(f"正在启动服务：")
        logger.info(f"如需查看 llm_api 日志，请前往 {Settings.basic_settings.LOG_PATH}")

    processes = {}

    def process_count():
        return len(processes)

    api_started = manager.Event() #这行代码在主进程中创建了一个共享事件 api_started，然后将它作为参数传给了子进程。
    if args.api:
        process = Process(
            target=run_api_server,
            name=f"API Server",
            kwargs=dict(
                started_event=api_started, #将共享事件 api_started作为参数传给了子进程。 这意味着子进程（运行 run_api_server 函数的那个新进程）也能访问这个 Event 对象。
                run_mode=run_mode,
            ),
            daemon=False,
        )
        processes["api"] = process

    webui_started = manager.Event()
    if args.webui:
        process = Process(
            target=run_webui,
            name=f"WEBUI Server",
            kwargs=dict(
                started_event=webui_started,
                run_mode=run_mode,
            ),
            daemon=True,
        )
        processes["webui"] = process

    try:
        if p := processes.get("api"):
            p.start() # 启动api.py进程，它会开启一个全新的python进程，在这个进程中执行run_api_server函数，主进程继续往下走（但最后会等子进程执行完再接着走后续的逻辑）
            p.name = f"{p.name} ({p.pid})"
            api_started.wait()  # 等待api.py启动完成

        if p := processes.get("webui"):
            p.start() # 与上面同理，创建一个新的进程来执行
            p.name = f"{p.name} ({p.pid})"
            webui_started.wait()  # 等待webui.py启动完成

        dump_server_info(after_start=True, args=args)

        # 等待所有进程退出
        while processes:
            for p in processes.values():
                p.join(2)
                if not p.is_alive():
                    processes.pop(p.name)
    except Exception as e:
        logger.error(e)
        logger.warning("Caught KeyboardInterrupt! Setting stop event...")
    finally:
        for p in processes.values():
            logger.warning("Sending SIGKILL to %s", p)
            # Queues and other inter-process communication primitives can break when
            # process is killed, but we don't care here

            if isinstance(p, dict):
                for process in p.values():
                    process.kill()
            else:
                p.kill()

        for p in processes.values():
            logger.info("Process status: %s", p)


@click.command(help="启动服务")
@click.option(
    "-a",
    "--all",
    "all",
    is_flag=True,
    default=True,
    help="run api.py and webui.py",
)
@click.option(
    "--api",
    "api",
    is_flag=True,
    help="run api.py",
)
@click.option(
    "-w",
    "--webui",
    "webui",
    is_flag=True,
    help="run webui.py server",
)
def main(all, api, webui):
    class args: # 定义了一个名为 args 的内部类，该类用于存储脚本运行时的参数
        ...
    args.all = all # 将传入的参数值赋给 args 类的实例变量。
    args.api = api #
    args.webui = webui

    # 添加这行代码
    cwd = os.getcwd() # 获取当前工作目录的路径，并将其存储在变量 cwd 中。
    sys.path.append(cwd) # 将当前工作目录的路径添加到Python的模块搜索路径中，这样就可以导入当前目录下的模块。
    mp.freeze_support() # 调用 multiprocessing 模块的 freeze_support() 函数，这通常用于支持冻结多进程程序（例如，创建可执行文件）。
    print("cwd:" + cwd)
    from chatchat.server.knowledge_base.migrate import create_tables
    create_tables()

    if sys.version_info < (3, 10):
        loop = asyncio.get_event_loop()
    else:
        try:
            loop = asyncio.get_running_loop() # 尝试使用 asyncio.get_running_loop() 获取当前运行的事件循环。
        except RuntimeError: # 如果当前没有运行的事件循环，捕获 RuntimeError 异常,并创建一个新的事件循环。
            loop = asyncio.new_event_loop()

        asyncio.set_event_loop(loop) # 使用 asyncio.set_event_loop(loop) 设置当前线程的事件循环。
    loop.run_until_complete(start_main_server(args)) # 使用事件循环 loop 运行 start_main_server 协程，并传递 args 对象作为参数。这个协程可能负责启动主服务器。


if __name__ == "__main__":
    main()
