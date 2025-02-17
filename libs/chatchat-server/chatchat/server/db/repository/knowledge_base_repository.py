from chatchat.server.db.models.knowledge_base_model import (
    KnowledgeBaseModel,
    KnowledgeBaseSchema,
)
from chatchat.server.db.session import with_session


@with_session
def add_kb_to_db(session, kb_name, kb_info, vs_type, embed_model):
    '''添加或更新知识库到数据库。如果已经存在同名知识库，则更新现有知识库，否则创建一个新的知识库'''
    kb = (
        session.query(KnowledgeBaseModel)
        .filter(KnowledgeBaseModel.kb_name.ilike(kb_name))
        .first()
    )
    if not kb:
        kb = KnowledgeBaseModel(
            kb_name=kb_name, kb_info=kb_info, vs_type=vs_type, embed_model=embed_model
        )
        session.add(kb)
    else:  # update kb with new vs_type and embed_model
        kb.kb_info = kb_info
        kb.vs_type = vs_type
        kb.embed_model = embed_model
    return True


@with_session
def list_kbs_from_db(session, min_file_count: int = -1):
    '''列出文件数量大于给定最小值的所有知识库。'''
    kbs = (
        session.query(KnowledgeBaseModel) # 使用 SQLAlchemy 的 ORM 查询接口 session.query(KnowledgeBaseModel) 来构建查询。
        .filter(KnowledgeBaseModel.file_count > min_file_count) # 只选择文件数量大于 min_file_count 的知识库。
        .all() # 执行查询并获取所有匹配记录。
    )
    kbs = [KnowledgeBaseSchema.model_validate(kb) for kb in kbs] # 每个从数据库中取出的知识库模型对象 kb，使用 KnowledgeBaseSchema.model_validate(kb) 将其转换为 Pydantic 模型实例。这一步骤有助于验证和序列化数据。
    return kbs


@with_session
def kb_exists(session, kb_name):
    '''检查指定名称的知识库是否存在。'''
    kb = (
        session.query(KnowledgeBaseModel)
        .filter(KnowledgeBaseModel.kb_name.ilike(kb_name))
        .first()
    )
    status = True if kb else False
    return status


@with_session
def load_kb_from_db(session, kb_name):
    '''从数据库加载特定名称的知识库信息。'''
    kb = (
        session.query(KnowledgeBaseModel)
        .filter(KnowledgeBaseModel.kb_name.ilike(kb_name))
        .first()
    )
    if kb:
        kb_name, vs_type, embed_model = kb.kb_name, kb.vs_type, kb.embed_model
    else:
        kb_name, vs_type, embed_model = None, None, None
    return kb_name, vs_type, embed_model


@with_session
def delete_kb_from_db(session, kb_name):
    '''删除指定名称的知识库。'''
    kb = (
        session.query(KnowledgeBaseModel)
        .filter(KnowledgeBaseModel.kb_name.ilike(kb_name))
        .first()
    )
    if kb:
        session.delete(kb)
    return True


@with_session
def get_kb_detail(session, kb_name: str) -> dict:
    '''获取指定名称的知识库详情。'''
    kb: KnowledgeBaseModel = (
        session.query(KnowledgeBaseModel)
        .filter(KnowledgeBaseModel.kb_name.ilike(kb_name))
        .first()
    )
    if kb:
        return {
            "kb_name": kb.kb_name,
            "kb_info": kb.kb_info,
            "vs_type": kb.vs_type,
            "embed_model": kb.embed_model,
            "file_count": kb.file_count,
            "create_time": kb.create_time,
        }
    else:
        return {}
