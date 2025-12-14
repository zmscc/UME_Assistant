"""
:return: (is_handled_by_slot, response_or_new_query, metadata)
para1:是否由槽位填充（任务型对话）模型处理了。
para2:如果是任务型对话，可能是追问（如“请问您要订哪天的票？”）；如果不是，则原样返回 query。
para3:附加信息，如 "[SLOT_FILLING]" 表示正在填槽，"[SLOT_COMPLETED]" 表示槽已填完。
"""
# slot_middleware.py

from typing import Any, Tuple

from .session_manager import append_message, get_chatbot, save_chatbot, clear_session
from .chatbot_model import ChatbotModel
from .utils import load_all_scene_configs


async def process_with_slot_model(
    query: str,
    conversation_id: str,
    model: str = "your-model-name",
    stream: bool = False
) -> Tuple[bool, str, str]:
    append_message(conversation_id, "user", query)

    if any(w in query for w in ["取消", "退出", "结束"]):
        clear_session(conversation_id)
        cancel_msg = "已取消当前操作。"
        append_message(conversation_id, "assistant", cancel_msg)
        return True, cancel_msg, "[CANCEL]"

    bot = get_chatbot(conversation_id)
    templates = load_all_scene_configs()
    if not bot:
        bot = ChatbotModel(templates)

    result, is_complete = bot.process_multi_question(query, conversation_id)

    # ✅ 优先处理 【FINAL】
    if isinstance(result, str) and result.strip().startswith("【FINAL】"):
        final_content = result[len("【FINAL】"):].strip()
        append_message(conversation_id, "assistant", final_content)
        clear_session(conversation_id)
        return True, final_content, "[GREETING]"

    # 保存状态（如果有 purpose）
    if bot.current_purpose:
        save_chatbot(conversation_id, bot)

        if not result or not result.strip():
            clear_session(conversation_id)
            return True, "抱歉，处理请求时出错。", "[ERROR]"

        if is_complete:
            # ✅ 关键修复：这不是最终答案，要走 RAG！
            return False, result, "[SLOT_COMPLETED]"  # ← is_final=False !
        else:
            append_message(conversation_id, "assistant", result)
            return False, result, "[NEED_MORE]"
    else:
        return False, query, "[NO_SCENE]"