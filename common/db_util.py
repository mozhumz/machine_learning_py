import pymysql
from dbutils.pooled_db import PooledDB
POOL = PooledDB(
    # 使用链接数据库的模块
    creator=pymysql,
    # 连接池允许的最大连接数，0和None表示不限制连接数
    maxconnections=6,
    # 初始化时，链接池中至少创建的空闲的链接，0表示不创建
    mincached=2,
    # 链接池中最多闲置的链接，0和None不限制
    maxcached=5,
    # 链接池中最多共享的链接数量，0和None表示全部共享。
    # 因为pymysql和MySQLdb等模块的 threadsafety都为1，
    # 所有值无论设置为多少，maxcached永远为0，所以永远是所有链接都共享。
    maxshared=3,
    # 连接池中如果没有可用连接后，是否阻塞等待。True，等待；False，不等待然后报错
    blocking=True,
    # 一个链接最多被重复使用的次数，None表示无限制
    maxusage=None,
    # 开始会话前执行的命令列表。如：["set datestyle to ...", "set time zone ..."]
    setsession=[],
    # ping MySQL服务端，检查是否服务可用。
    #  如：0 = None = never, 1 = default = whenever it is requested,
    # 2 = when a cursor is created, 4 = when a query is executed, 7 = always
    ping=0,
    # 主机地址
    host='127.0.0.1',
    # 端口
    port=3306,
    # 数据库用户名
    user='root',
    # 数据库密码
    password='123456',
    # 数据库名
    database='bigdata',
    # database='mysql',
    # 字符编码
    charset='utf8'
)

def getConn():
    # 检测当前正在运行连接数的是否小于最大链接数，如果不小于则：等待或报raise TooManyConnections异常
    # 否则则优先去初始化时创建的链接中获取链接 SteadyDBConnection。
    # 然后将SteadyDBConnection对象封装到PooledDedicatedDBConnection中并返回。
    # 如果最开始创建的链接没有链接，则去创建一个SteadyDBConnection对象，再封装到PooledDedicatedDBConnection中并返回。
    # 一旦关闭链接后，连接就返回到连接池让后续线程继续使用。

    # 创建连接,POOL数据库连接池中
    conn = POOL.connection()

    # 创建游标
    # cursor = conn.cursor()
    # # SQL语句
    # cursor.execute('select * from tb1')
    # # 执行结果
    # result = cursor.fetchall()
    # # 将conn释放,放回连接池
    # conn.close()
    return conn

# if __name__ == '__main__':
#     print('start')
#     conn=getConn()
#     cursor = conn.cursor()
#     cursor.execute('select * from db')
#     res=cursor.fetchall()
#     print(res)
#     conn.close()