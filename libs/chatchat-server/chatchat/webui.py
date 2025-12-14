import sys
import streamlit as st
import streamlit_antd_components as sac

from chatchat import __version__
from chatchat.server.utils import api_address
from chatchat.webui_pages.dialogue.dialogue import dialogue_page
from chatchat.webui_pages.kb_chat import kb_chat
from chatchat.webui_pages.knowledge_base.knowledge_base import knowledge_base_page
from chatchat.webui_pages.utils import *

api = ApiRequest(base_url=api_address())

if __name__ == "__main__":
    is_lite = "lite" in sys.argv  # TODO: remove lite mode

    st.set_page_config(
        "航旅助手 WebUI",
        get_img_base64("chatchat_icon_blue_square_v2.png"),
        initial_sidebar_state="expanded",
        menu_items={
            "About": f"""欢迎使用 航旅助手 WebUI {__version__}！""",
        },
        layout="centered",
    )

    # ================================
    # ✅ 安全获取背景图 Base64（防空值/异常）
    # ================================
    try:
        # 尝试加载图片，如果失败，则使用空字符串
        BACKGROUND_IMAGE_BASE64 = get_img_base64("C:/UME_Assistant/libs/chatchat-server/chatchat/img/imageTest.png")
    except Exception as e:
        # print(f"Error loading background image: {e}")
        BACKGROUND_IMAGE_BASE64 = ""

    # ================================
    # ✅ 构建背景图 CSS（仅当有图片时启用）
    # ================================
    # 注意：这里我们仅将背景图逻辑注入到 .main 中，并使用 !important 提高优先级
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
    # ✅ 全局 CSS：侧边栏颜色已加深
    # =========================================================================
    css = f"""
    <style>
    /* 1. 主页面背景色 (保持: #E0F2FF) */
    .main {{ background-color: #E0F2FF; }}

    /* 2. 🔥 侧边栏背景 (已加深: #A3D5FF) */
    [data-testid="stSidebarContent"] {{ 
        background-color: #A3D5FF; 
        padding-top: 20px;
    }}

    /* 3. 您原有的内边距优化 */
    [data-testid="stSidebarUserContent"] {{
        padding-top: 20px;
    }}
    .block-container {{
        padding-top: 25px !important;
        padding-bottom: 0 !important;
    }}

    /* 🔥 全局背景：设为回退颜色 */
    html, body, .stApp {{
        height: 100%;
        margin: 0;
        padding: 0;
        background-color: #E0F2FF; /* 与主背景一致 */
        overflow-x: hidden; /* 防止水平滚动 */
    }}

    /* 🔥 主内容区：设置回退颜色（如果图片加载失败）和最小高度 */
    .main {{
        background-color: #E0F2FF;
        min-height: 100vh;
        padding: 0;
        margin: 0;
    }}


    /* 🔥 ⭐ 关键修复：底部固定区域背景色 */
    div[data-testid="stBottomBlockContainer"] {{
        background-color: #E0F2FF !important; /* 与主背景一致 */
        padding-bottom: 0 !important;
        margin-bottom: 0 !important;
    }}

    /* 🔥 ⭐ 关键修复：底部容器的父级（防止白边） */
    div.st-emotion-cache-uhkwx6 {{ 
        background-color: #E0F2FF !important; /* 与主背景一致 */
    }}

    /* 🔥 隐藏默认 header 和 footer */
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

    /* 🔥 禁用滚动条（防布局抖动） */
    ::-webkit-scrollbar {{
        display: none;
    }}
    body {{
        -ms-overflow-style: none;  /* IE and Edge */
        scrollbar-width: none;     /* Firefox */
    }}

    /* 🔥 ⭐ 背景图支持：使用 !important 覆盖纯色背景 */
    {bg_image_css}
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