import sys
import streamlit as st
import streamlit_antd_components as sac

from chatchat import __version__
from chatchat.server.utils import api_address
from chatchat.webui_pages.kb_chat import kb_chat
from chatchat.webui_pages.knowledge_base.knowledge_base import knowledge_base_page
from chatchat.webui_pages.utils import *

# 确保 ApiRequest 和相关函数（如 api_address）是可用的
api = ApiRequest(base_url=api_address())

if __name__ == "__main__":
    is_lite = "lite" in sys.argv

    st.set_page_config(
        "航旅助手 WebUI",
        get_img_base64("chatchat_icon_blue_square_v2.png"),  # 假设这里的路径是正确的
        initial_sidebar_state="expanded",
        menu_items={
            "About": f"""欢迎使用 航旅助手 WebUI {__version__}！""",
        },
        layout="centered",
    )

    # ================================
    # 1. 背景图处理
    # ================================
    # 尝试加载图片，如果失败，则使用空字符串，避免程序崩溃
    BACKGROUND_IMAGE_BASE64 = ""
    try:
        # 假设该函数和路径是正确的
        BACKGROUND_IMAGE_BASE64 = get_img_base64("C:/UME_Assistant/libs/chatchat-server/chatchat/img/imageTest.png")
    except Exception:
        # 如果加载失败，保持 BACKGROUND_IMAGE_BASE64 为空
        pass

    # 构建背景图 CSS
    bg_image_css = ""
    if BACKGROUND_IMAGE_BASE64:
        bg_image_css = f"""
        .main {{
            background-image: url("data:image/png;base64,{BACKGROUND_IMAGE_BASE64}") !important;
            background-size: cover !important;
            background-repeat: no-repeat !important;
            background-attachment: fixed !important;
            background-position: center !important;
        }}
        """

    # ================================
    # 2. 注入全局 CSS 和对话对齐修复 (最终确定版：基于 DOM 结构和 stChatMessageContent 的 role 属性)
    # =========================================================================
    css = f"""
    <style>
    /* 全局样式 */
    .main {{ background-color: #E0F2FF; }}
    [data-testid="stSidebarContent"] {{ background-color: #A3D5FF; padding-top: 20px; }}

    /* 背景图支持 */
    {bg_image_css}

    /* ========================================================= */
    /* ⭐⭐⭐ 最终确定修复：基于 stChatMessageContent 上的 role="user" 属性 ⭐⭐⭐ */
    /* --------------------------------------------------------- */

    /* 1. 强制用户消息（User Message）容器反向排列并靠右 */
    /* 注意：我们将通过 content 上的 role="user" 来定位消息 */
    div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageContent"][aria-label*="user"]) {{
        /* 强制最外层 Flexbox 反向 */
        display: flex !important;
        flex-direction: row-reverse !important; 
        justify-content: flex-start !important; /* 确保整个容器靠右 */
        margin-right: 1rem !important; 
        margin-left: 0 !important;
        padding-left: 0 !important;
    }}

    /* 1b. 调整用户头像和消息内容之间的间距（反转后的间距）*/
    /* 头像在最右侧，消息内容在其左侧 */
    div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageContent"][aria-label*="user"]) > div:first-child {{
        margin-left: 0.5rem !important;
        margin-right: 0 !important;
    }}

    /* 2. 强制用户消息内容 (stChatMessageContent) 的容器靠右对齐 */
    div[data-testid="stChatMessageContent"][aria-label*="user"] {{
        /* 强制内容块的父容器靠右对齐 */
        display: flex !important;
        flex-direction: column !important;
        align-items: flex-end !important; /* 核心：让气泡内容靠右对齐 */
        flex-grow: 1 !important; 
        width: 100% !important;
        /* 清除可能存在的 text-align: left */
        text-align: right !important;
    }}

    /* 3. 覆盖气泡本身及其所有内部块的样式 */
    div[data-testid="stChatMessageContent"][aria-label*="user"] [data-testid="stVerticalBlock"],
    div[data-testid="stChatMessageContent"][aria-label*="user"] [data-testid="stVerticalBlock"] > div,
    div[data-testid="stChatMessageContent"][aria-label*="user"] [data-testid="stMarkdownContainer"], 
    div[data-testid="stChatMessageContent"][aria-label*="user"] [data-testid="stMarkdownContainer"] p {{
        text-align: right !important; /* 文本强制靠右 */
        width: 100% !important;
        align-items: flex-end !important; /* 确保内部 Flex 块靠右 */
        margin-left: auto !important; /* 将整个气泡推到右侧 */
        margin-right: 0 !important; 
    }}

    /* --------------------------------------------------------- */
    /* 4. 确保助手消息（Assistant Message）保持默认靠左，且不受影响 */
    /* 我们使用 aria-label*="assistant" 来保证选择器的专一性 */
    /* --------------------------------------------------------- */
    div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageContent"][aria-label*="assistant"]) {{
        /* 强制助手消息容器为正向排列和左对齐 */
        display: flex !important;
        flex-direction: row !important;
        justify-content: flex-start !important;
        margin-left: 1rem !important; 
        margin-right: 0 !important;
        padding-right: 0 !important;
    }}

    div[data-testid="stChatMessageContent"][aria-label*="assistant"],
    div[data-testid="stChatMessageContent"][aria-label*="assistant"] [data-testid="stVerticalBlock"],
    div[data-testid="stChatMessageContent"][aria-label*="assistant"] [data-testid="stVerticalBlock"] > div,
    div[data-testid="stChatMessageContent"][aria-label*="assistant"] [data-testid="stMarkdownContainer"], 
    div[data-testid="stChatMessageContent"][aria-label*="assistant"] [data-testid="stMarkdownContainer"] p {{
        text-align: left !important;
        width: 100% !important;
        align-items: flex-start !important;
        margin-right: auto !important; 
        margin-left: 0 !important;
    }}

    /* ========================================================= */

    /* 您的其他样式... */
    .block-container {{
        padding-top: 25px !important;
        padding-bottom: 0 !important;
    }}
    [data-testid="stSidebarUserContent"] {{
        padding-top: 20px;
    }}
    html, body, .stApp {{
        height: 100%;
        margin: 0;
        padding: 0;
        background-color: #E0F2FF;
        overflow-x: hidden;
    }}
    .main {{
        background-color: #E0F2FF;
        min-height: 100vh;
        padding: 0;
        margin: 0;
    }}
    div[data-testid="stBottomBlockContainer"] {{
        background-color: #E0F2FF !important;
        padding-bottom: 0 !important;
        margin-bottom: 0 !important;
    }}
    div.st-emotion-cache-uhkwx6 {{ 
        background-color: #E0F2FF !important;
    }}
    header[data-testid="stHeader"] {{
        background-color: transparent !important;
        height: 0 !important;
        padding: 0 !important;
    }}
    footer[data-testid="stFooter"] {{
        visibility: hidden !important;
        height: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
    }}
    ::-webkit-scrollbar {{
        display: none;
    }}
    body {{
        -ms-overflow-style: none;
        scrollbar-width: none;
    }}

    </style>
    """

    st.markdown(css, unsafe_allow_html=True)

    # ================================
    # 侧边栏菜单
    # ================================
    with st.sidebar:
        st.image(get_img_base64("logo-long-chatchat-trans-v2.png"), use_column_width=True)
        st.caption(f"""<p align="right">当前版本：{__version__}</p>""", unsafe_allow_html=True)

        selected_page = sac.menu(
            [
                sac.MenuItem("对话管理", icon="database"),
                sac.MenuItem("知识库管理", icon="hdd-stack"),
            ],
            key="selected_page",
            open_index=0,
        )

        sac.divider()

    # 页面路由
    if selected_page == "知识库管理":
        knowledge_base_page(api=api, is_lite=is_lite)
    elif selected_page == "对话管理":
        kb_chat(api=api)
