import pandas as pd
import math
import numpy as np
df = pd.DataFrame({'Animal': ['Falcon', 'Falcon',
                              'Parrot', 'Parrot'],
                   'Max Speed': [380., 370., 24., 26.],
                   'Col_c':['c1','c2','c3','c4']
                   })
df1 = pd.DataFrame({'Animal': ['Falcon-df1', 'Falcon-df1',
                              'Parrot-df1', 'Parrot-df1'],
                   'id':[1,2,3,4]
                   })
df2 = pd.DataFrame({'Animal2': ['Falcon-df2', 'Falcon-df2',
                               'Parrot-df2', 'Parrot-df2'],
                    'id':[1,2,3,4]
                    })

df1['id']=df1['id'].apply(lambda x:str(x)+'_')
df2['id']=df2['id'].apply(lambda x:str(x)+'_')
df1=pd.merge(df1,df2,on='id')
print(df1)

# df1.set_index('id',drop=False,inplace=True)
# df2.set_index('id',drop=False,inplace=True)
# print(df1)
# df1.id=df1.id.astype(np.int32)
# # df1的id去匹配df2的index
# df1['Animal2']=df1.id.map(df2.Animal2)
print(df1.dtypes)

#
# # print(df)
#
# df_g=df.groupby(['Animal']).agg(list).rename(columns={'Max Speed':'Max Speed-arr','Col-c':'Col-c-arr'})
# print(df_g)
#
# a=[1,2]
# b=[1,2]
# # 引用比较
# print(a is b)
# # 内容比较
# print(a==b)
#
# def getHAD(a_list,total):
#     res=.0
#     for i in a_list:
#         res-=i/total*math.log2(i/total)
#     return res
# total=100.
# a_list=[50,50]
# print(getHAD(a_list,total))
# print('-------------------')
# a_list=[50,40,10]
# print(getHAD(a_list,total))
# order_id='1'
# labels=[]
# user_products=['1','2','3']
# train=['2','3']
# labels += [(order_id, product) in train for product in user_products]
# print(labels)
# arr1=np.array(["1","2","3"],dtype=np.str)
# print(arr1)
# print('abc'+(arr1[0]))

# print(df)
# df.set_index('Col_c',drop=False,inplace=True)
# print(df[0])
# for idx,row in df.iterrows():
#     print(row['Col-c'])

# t1=(5,-1,2)
# t2=(2,1,3)
# print(max(t1,t2))

# print(df.Animal['c1'])

# df1=pd.DataFrame({'id':[1,2,3],'name':['Andy1','Jacky1','Bruce1']})
# df2=pd.DataFrame({'id':[1,2],'name':['Andy2','Jacky2']})
#
# s = df2.set_index('id')['name']
# print('df1',df1)
# print('----------------')
# print('s',s)
# print('----------------')
# # df1['name'] = df1['id'].map(s).fillna(df1['name']).astype(str)
# print(df1['id'].map(s))
# print('----------------')
# print(df1['id'].map(s).fillna(df1['name']).astype(str))

# df = pd.DataFrame(np.random.randn(10000, 4))
#
# df.iloc[:9998] = np.nan
#
# sdf = df.astype(pd.SparseDtype("float", np.nan))
#
# sdf.head()

f_to_use = ['user_total_orders', 'user_total_items', 'total_distinct_items',
            'user_average_days_between_orders', 'user_average_basket',
            'order_hour_of_day', 'days_since_prior_order', 'days_since_ratio',
            'aisle_id', 'department_id', 'product_orders', 'product_reorders',
            'product_reorder_rate', 'UP_orders', 'UP_orders_ratio',
            'UP_average_pos_in_cart', 'UP_orders_since_last',
            'UP_delta_hour_vs_last']
# 18
import tensorflow as tf
est = tf.estimator.BoostedTreesRegressor(f_to_use,n_batches_per_layer=1)

print(len(f_to_use))
