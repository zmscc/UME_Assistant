from sqlalchemy import create_engine

engine = create_engine("mysql+pymysql://root:123456@192.168.200.130:3306/assistant")
try:
    conn = engine.connect()
    print("连接成功！")
    conn.close()
except Exception as e:
    print("连接失败:", e)