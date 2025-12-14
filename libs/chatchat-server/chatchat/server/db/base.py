import json

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import DeclarativeMeta, declarative_base
from sqlalchemy.orm import sessionmaker

from chatchat.settings import Settings


# engine = create_engine(
#     Settings.basic_settings.SQLALCHEMY_DATABASE_URI, # SQLAlchemy
#     json_serializer=lambda obj: json.dumps(obj, ensure_ascii=False),
# )
#
# SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
#
# Base: DeclarativeMeta = declarative_base() # 创建对象的基类:


# 假设你的 MySQL 用户名是 'myuser', 密码是 'mypass', 数据库名是 'mydb'
SQLALCHEMY_DATABASE_URI = "mysql+pymysql://root:123456@192.168.200.130:3306/Assistant"

engine = create_engine(
    SQLALCHEMY_DATABASE_URI,
    json_serializer=lambda obj: json.dumps(obj, ensure_ascii=False),
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
session = SessionLocal()

Base: DeclarativeMeta = declarative_base()  #  这行代码是在声明 Base 是一个由 DeclarativeMeta 元类创建的类。这样，任何继承自 Base 的类都会自动获得声明式映射的功能。

