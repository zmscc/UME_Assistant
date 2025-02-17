import uuid

from chatchat.server.db.models.conversation_model import ConversationModel
from chatchat.server.db.session import with_session

'''这个文件夹里的5个repository是对保存到数据库里的表做具体的操作'''
@with_session
def add_conversation_to_db(session, chat_type, name="", conversation_id=None):
    """
    新增聊天记录
    """
    if not conversation_id:
        conversation_id = uuid.uuid4().hex
    c = ConversationModel(id=conversation_id, chat_type=chat_type, name=name)

    session.add(c)
    return c.id
