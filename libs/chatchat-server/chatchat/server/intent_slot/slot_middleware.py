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

    # 处理取消指令
    if any(w in query for w in ["取消", "退出", "结束"]):
        clear_session(conversation_id)
        cancel_msg = "已取消当前操作。"
        append_message(conversation_id, "assistant", cancel_msg)
        return True, cancel_msg, "[CANCEL]"

    bot = get_chatbot(conversation_id)
    templates = load_all_scene_configs()
    if not bot:
        bot = ChatbotModel(templates)

    result = bot.process_multi_question(query, conversation_id)

    if bot.current_purpose:
        save_chatbot(conversation_id, bot)

        # 检查是否为【FINAL】标记的最终响应（如 greeting）
        if isinstance(result, str) and result.strip().startswith("【FINAL】"):
            final_content = result[len("【FINAL】"):].strip()
            append_message(conversation_id, "assistant", final_content)
            # 判断具体类型（可扩展）
            if "南航智能客服" in final_content:
                meta_tag = "[GREETING]"
            else:
                meta_tag = "[SLOT_COMPLETED]"
            return True, final_content, meta_tag

# todo 这里有问题
        # 如果 result 是追问（不以"根据以下需求"开头的字符串）
        if isinstance(result, str) and result.strip() and not result.strip().startswith("根据以下需求"):
            append_message(conversation_id, "assistant", result)
            return False, result, "[NEED_MORE]"  # 需要更多信息，不进 RAG

        # 如果槽位已完成（result 是完整问句）
        elif isinstance(result, str) and result.strip():
            return True, result, "[SLOT_COMPLETED]"

        else:
            # 异常情况
            clear_session(conversation_id)
            return True, "抱歉，处理请求时出错。", "[ERROR]"
    else:
        # 未命中任何场景 → 透传原始 query，标记为 [NO_SCENE]
        return False, query, "[NO_SCENE]"
