from hdfs.client import Client
import pandas as pd

def read_hdfs_by_client():
    '''
    方法一：使用hdfs库读取HDFS文件
    在读取数据时，要加上 encoding='utf-8'，否则字符串前面会有b'xxx'
    先写入list，再转为df，注意要对数据进行分列，最后要对指定字段转换数据类型
    :return:
    '''
    client = Client("http://hadoop-1-1:50070")
    lines = []
    with client.read("/user/spark/H2O/Wholesale_customers_data.csv", encoding='utf-8') as reader:
        for line in reader:
            lines.append(line.strip())
    column_str = lines[0]
    column_list = column_str.split(',')
    data = {"item_list": lines[1:]}

    df = pd.DataFrame(data=data)
    df[column_list] = df["item_list"].apply(lambda x: pd.Series([i for i in x.split(",")]))  ##重新指定列
    df.drop("item_list", axis=1, inplace=True)  ##删除列
    df.dtypes
    """
    Region              object
    Fresh               object
    Milk                object
    Grocery             object
    Frozen              object
    Detergents_Paper    object
    Delicassen          object
    target              object
    dtype: object
    """
    df = df.astype('int')  ##将object类型转为int64
    df.dtypes
    """
    Region              int64
    Fresh               int64
    Milk                int64
    Grocery             int64
    Frozen              int64
    Detergents_Paper    int64
    Delicassen          int64
    target              int64
    dtype: object
    """

import pydoop.hdfs as hdfs

lines = []
with hdfs.open('/user/spark/security/iris.csv', 'rt') as f:
    for line in f:
        ##print(line)
        lines.append(line.strip())


column_list = ['Sepal_Length', 'Sepal_Width', 'Petal_Length', 'Petal_Width', 'Species']

data = {"item_list":lines[0:]}

df = pd.DataFrame(data=data)
df[column_list] =  df["item_list"].apply(lambda x: pd.Series([i for i in x.split(",")]))  ##重新指定列
df.drop("item_list", axis=1, inplace=True)  ##删除列

##调整数据类型
df[['Sepal_Length', 'Sepal_Width', 'Petal_Length', 'Petal_Width']] = df[['Sepal_Length', 'Sepal_Width', 'Petal_Length', 'Petal_Width']].astype('float64')

df.dtypes
"""
Sepal_Length    float64
Sepal_Width     float64
Petal_Length    float64
Petal_Width     float64
Species          object
dtype: object
"""
