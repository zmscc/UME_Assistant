from contextlib import contextmanager
from functools import wraps

from sqlalchemy.orm import Session

from chatchat.server.db.base import SessionLocal

'''上下文管理器，负责创建和关闭数据库会话'''
@contextmanager
def session_scope() -> Session:
    """上下文管理器用于自动获取 Session, 避免错误"""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except:
        session.rollback()
        raise
    finally:
        session.close()

'''使用装饰器 @with_session 来管理数据库会话（session），并确保在函数执行期间正确处理事务。@with_session 装饰器：相当于 Spring 的 @Transactional'''
def with_session(f): # with_session 确保每次调用被装饰的函数时都会有一个新的数据库会话，并且无论函数是否成功完成，都会正确地提交或回滚事务。
    @wraps(f) # 这是一个来自 functools 模块的装饰器，用于保留原始函数的元数据（如名称、文档字符串等）。
    def wrapper(*args, **kwargs): # 这是实际包裹原函数的新函数。它接受任意数量的位置参数和关键字参数，并将它们传递给被装饰的函数。
        with session_scope() as session:
            try: # 如果函数执行没有抛出异常，session.commit() 提交事务，并返回函数的结果。
                result = f(session, *args, **kwargs)
                session.commit()
                return result
            except: # 如果发生异常，session.rollback() 回滚事务，然后重新抛出异常以允许调用者处理错误。
                session.rollback()
                raise

    return wrapper


def get_db() -> SessionLocal:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_db0() -> SessionLocal:
    db = SessionLocal()
    return db
