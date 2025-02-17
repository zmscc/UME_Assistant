'''这个脚本提供了一个接口，允许用户对聊天记录进行评分和提供反馈理由，并将这些信息保存到数据库中。这是聊天机器人服务中用户反馈机制的一部分。'''
from fastapi import Body

from chatchat.utils import build_logger
from chatchat.server.db.repository import feedback_message_to_db
from chatchat.server.utils import BaseResponse

logger = build_logger()


def chat_feedback(
    message_id: str = Body("", max_length=32, description="聊天记录id"),
    score: int = Body(0, max=100, description="用户评分，满分100，越大表示评价越高"),
    reason: str = Body("", description="用户评分理由，比如不符合事实等"),
):
    try:
        feedback_message_to_db(message_id, score, reason)
    except Exception as e:
        msg = f"反馈聊天记录出错： {e}"
        logger.error(f"{e.__class__.__name__}: {msg}")
        return BaseResponse(code=500, msg=msg)

    return BaseResponse(code=200, msg=f"已反馈聊天记录 {message_id}")
