# session_manager.py

import time
from typing import Dict, List, Optional
from .utils import load_all_scene_configs  # ✅ 正确导入


_SESSIONS: Dict[str, Dict] = {}
_TIMEOUT = 1800


def append_message(session_id: str, role: str, content: str):
    if session_id not in _SESSIONS:
        _SESSIONS[session_id] = {"data": None, "messages": [], "ts": time.time()}
    messages = _SESSIONS[session_id]["messages"]
    messages.append({"role": role, "content": content})
    if len(messages) > 10:
        messages.pop(0)
    _SESSIONS[session_id]["ts"] = time.time()


def get_conversation_history(session_id: str) -> List[Dict[str, str]]:
    now = time.time()
    expired = [k for k, v in _SESSIONS.items() if now - v["ts"] > _TIMEOUT]
    for k in expired:
        del _SESSIONS[k]
    return _SESSIONS.get(session_id, {}).get("messages", [])


def get_chatbot(session_id: str) -> Optional['ChatbotModel']:
    now = time.time()
    expired = [k for k, v in _SESSIONS.items() if now - v["ts"] > _TIMEOUT]
    for k in expired:
        del _SESSIONS[k]

    if session_id in _SESSIONS and _SESSIONS[session_id]["data"] is not None:
        from .chatbot_model import ChatbotModel
        templates = load_all_scene_configs()  # ✅
        return ChatbotModel.from_dict(_SESSIONS[session_id]["data"], templates)
    return None


def save_chatbot(session_id: str, bot: 'ChatbotModel'):
    if session_id not in _SESSIONS:
        _SESSIONS[session_id] = {"data": None, "messages": [], "ts": time.time()}
    _SESSIONS[session_id]["data"] = bot.to_dict()
    _SESSIONS[session_id]["ts"] = time.time()


def clear_session(session_id: str):
    _SESSIONS.pop(session_id, None)