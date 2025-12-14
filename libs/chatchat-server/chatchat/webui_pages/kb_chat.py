from datetime import datetime
import uuid
import openai
import streamlit as st
from streamlit_chatbox import *
from streamlit_extras.bottom_container import bottom

# 确保 chatchat.settings.Settings 被导入
try:
    from chatchat.settings import Settings
except ImportError:
    # 临时兼容处理，防止Settings未被导入时报错
    class DummySettings:
        class model_settings:
            HISTORY_LEN = 5
            TEMPERATURE = 0.7

        class kb_settings:
            DEFAULT_KNOWLEDGE_BASE = "samples"
            VECTOR_SEARCH_TOP_K = 3
            SEARCH_ENGINE_TOP_K = 3
            SCORE_THRESHOLD = 0.8
            DEFAULT_SEARCH_ENGINE = "bing"


    Settings = DummySettings

from chatchat.server.utils import get_config_models, get_config_platforms, get_default_llm, api_address
from chatchat.webui_pages.dialogue.dialogue import (save_session, restore_session, rerun,
                                                    get_messages_history, upload_temp_docs,
                                                    add_conv, del_conv, clear_conv)
from chatchat.webui_pages.utils import *

chat_box = ChatBox(
    assistant_avatar=get_img_base64("chatchat_icon_blue_square_v2.png"),
    user_avatar="🙂"
)

'''初始化会话状态'''


def init_widgets():
    st.session_state.setdefault("history_len", Settings.model_settings.HISTORY_LEN)
    st.session_state.setdefault("selected_kb", Settings.kb_settings.DEFAULT_KNOWLEDGE_BASE)
    st.session_state.setdefault("kb_top_k", Settings.kb_settings.VECTOR_SEARCH_TOP_K)
    st.session_state.setdefault("se_top_k", Settings.kb_settings.SEARCH_ENGINE_TOP_K)
    st.session_state.setdefault("score_threshold", Settings.kb_settings.SCORE_THRESHOLD)
    st.session_state.setdefault("search_engine", Settings.kb_settings.DEFAULT_SEARCH_ENGINE)
    st.session_state.setdefault("return_direct", False)
    st.session_state.setdefault("cur_conv_name", chat_box.cur_chat_name)
    st.session_state.setdefault("last_conv_name", chat_box.cur_chat_name)
    st.session_state.setdefault("file_chat_id", None)


def kb_chat(api: ApiRequest):
    ctx = chat_box.context
    ctx.setdefault("uid", uuid.uuid4().hex)
    ctx.setdefault("file_chat_id", None)
    ctx.setdefault("llm_model", get_default_llm())
    ctx.setdefault("temperature", Settings.model_settings.TEMPERATURE)
    init_widgets()

    '''检查当前会话名称是否发生变化'''
    if st.session_state.cur_conv_name != st.session_state.last_conv_name:
        save_session(st.session_state.last_conv_name)
        restore_session(st.session_state.cur_conv_name)
        st.session_state.last_conv_name = st.session_state.cur_conv_name

    @st.experimental_dialog("模型配置", width="large")
    def llm_model_setting():
        # 模型
        cols = st.columns(3)
        platforms = ["所有"] + list(get_config_platforms())
        platform = cols[0].selectbox("选择模型平台", platforms, key="platform")
        llm_models = list(
            get_config_models(
                model_type="llm", platform_name=None if platform == "所有" else platform
            )
        )
        llm_models += list(
            get_config_models(
                model_type="image2text", platform_name=None if platform == "所有" else platform
            )
        )
        llm_model = cols[1].selectbox("选择LLM模型", llm_models, key="llm_model")
        temperature = cols[2].slider("Temperature", 0.0, 1.0, key="temperature")
        system_message = st.text_area("System Message:", key="system_message")
        if st.button("OK"):
            rerun()

    @st.experimental_dialog("重命名会话")
    def rename_conversation():
        name = st.text_input("会话名称")
        if st.button("OK"):
            chat_box.change_chat_name(name)
            restore_session()
            st.session_state["cur_conv_name"] = name
            rerun()

    # 配置参数
    with st.sidebar:
        st.subheader("RAG 配置")
        dialogue_mode = "知识库问答"  # 模式硬编码
        placeholder = st.empty()
        st.divider()

        prompt_name = "default"
        history_len = st.number_input("历史对话轮数：", 0, 20, key="history_len")
        kb_top_k = st.number_input("匹配知识条数：", 1, 20, key="kb_top_k")
        ## Bge 模型会超过1
        score_threshold = st.slider("知识匹配分数阈值：", 0.0, 2.0, step=0.01, key="score_threshold")
        return_direct = st.checkbox("仅返回检索结果", key="return_direct")

        selected_kb = None

        def on_kb_change():
            st.toast(f"已加载知识库： {st.session_state.selected_kb}")

        with placeholder.container():
            kb_list = [x["kb_name"] for x in api.list_knowledge_bases()]

            if not kb_list:
                st.warning("知识库列表为空，请先前往知识库管理页面创建或加载知识库！")
                selected_kb = None
            else:
                try:
                    default_index = kb_list.index(st.session_state.selected_kb)
                except:
                    default_index = 0

                selected_kb = st.selectbox(
                    "请选择知识库：",
                    kb_list,
                    index=default_index,
                    on_change=on_kb_change,
                    key="selected_kb",
                )

    # Display chat messages from history on app rerun
    chat_box.output_messages()
    chat_input_placeholder = "请输入对话内容，换行请使用Shift+Enter。"

    llm_model = ctx.get("llm_model")

    # chat input
    with bottom():
        cols = st.columns([1, 0.2, 15, 1])
        if cols[0].button(":gear:", help="模型配置"):
            widget_keys = ["platform", "llm_model", "temperature", "system_message"]
            chat_box.context_to_session(include=widget_keys)
            llm_model_setting()

        # ⭐ 恢复清空按钮
        if cols[-1].button(":wastebasket:", help="清空对话"):
            chat_box.reset_history()
            rerun()

        prompt = cols[2].chat_input(chat_input_placeholder, key="prompt")

    # 如果没有知识库被选中，则不进行RAG流程
    if not selected_kb:
        return

    if prompt:
        history = get_messages_history(ctx.get("history_len", 0))
        messages = history + [{"role": "user", "content": prompt}]
        chat_box.user_say(prompt)

        extra_body = dict(
            top_k=kb_top_k,
            score_threshold=score_threshold,
            temperature=ctx.get("temperature"),
            prompt_name=prompt_name,
            return_direct=return_direct,
        )

        api_url = api_address(is_public=True)

        chat_box.use_chat_name(st.session_state.cur_conv_name)
        client = openai.Client(base_url=f"{api_url}/knowledge_base/local_kb/{selected_kb}", api_key="NONE")

        chat_box.ai_say([
            Markdown("...", in_expander=True, title="知识库匹配结果", state="running", expanded=return_direct),
            f"正在查询知识库 `{selected_kb}` ...",
        ])

        text = ""
        first = True

        '''处理来自 AI 模型的流式响应，并实时更新聊天界面以显示模型生成的回答。'''
        try:
            for d in client.chat.completions.create(messages=messages, model=llm_model, stream=True,
                                                    extra_body=extra_body):
                if first:
                    docs = getattr(d, 'docs', []) if hasattr(d, 'docs') else []
                    if docs:
                        chat_box.update_msg("\n\n".join(docs), element_index=0, streaming=False, state="complete")
                    else:
                        chat_box.update_msg("未找到匹配的知识或文档。", element_index=0, streaming=False, state="complete")

                    chat_box.update_msg("", streaming=False)
                    first = False
                    continue

                content = d.choices[0].delta.content
                if content:
                    text += content
                    chat_box.update_msg(text.replace("\n", "\n\n"), streaming=True)

            chat_box.update_msg(text, streaming=False)

        except openai.APIError as e:
            error_msg = getattr(e, 'body', str(e))
            st.error(f"知识库查询 API 失败：{error_msg}")
            chat_box.update_msg(f"查询失败，API 返回错误：{error_msg}", streaming=False)
            chat_box.update_msg("查询失败，请检查配置和 API 服务。", element_index=0, streaming=False, state="complete")
        except Exception as e:
            error_msg = getattr(e, 'body', str(e)) if hasattr(e, 'body') else str(e)
            st.error(f"知识库查询发生未知错误：{error_msg}")
            chat_box.update_msg(f"查询失败，发生未知错误：{error_msg}", streaming=False)
            chat_box.update_msg("查询失败，请检查配置和 API 服务。", element_index=0, streaming=False, state="complete")