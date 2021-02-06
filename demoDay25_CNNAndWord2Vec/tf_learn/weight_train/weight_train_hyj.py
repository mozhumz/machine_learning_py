import pandas as pd
import numpy as np
import random
from demoDay25_CNNAndWord2Vec.tf_learn.weight_train.get_score_hyj import get_wikiid_score
from common.db_util import getConn
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from concurrent.futures import ThreadPoolExecutor
import tensorflow.keras as ks
import tensorflow as tf
import xgboost as xgb

def randint(a, b):
    return random.randint(a, b)

def gen_test_data():
        '''    生成测试数据    :return:    '''
        conn = getConn()
        # threadPool = ThreadPoolExecutor(max_workers=20)
        sqlPre = "insert into his_task_used_time (wkiid,match_rate, is_nine_trs, t1_distinct, t2_distinct, t3_distinct, t4_distinct, " \
                 "is_corr_nod,amount ,inx_roles ,inx_grpid," \
                 "yichu,fstflg,y)" \
                 "values(%s, %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s, %s,%s)"

        for i in range(1000):

            # threadPool.submit(add_batch, conn=conn, i=i, sqlPre=sqlPre)
            add_batch(conn,  i,  sqlPre)

        conn.close()


def add_batch(conn, i, sqlPre):
    data_dict={}
    for j in range(100):
        value = [0 for x in range(20)]
        match_rate, is_nine_trs, t1_distinct, t2_distinct, t3_distinct, t4_distinct, is_corr_nod, strorg, trschl, \
        ntsgrp, cusflg, paynum, drenum, imgcnt, amount, cnyno, inx_roles, inx_grpid, yichu, fstflg = value
        # 下标0-6  -6 -1~-4
        match_rate = randint(0, 100)
        is_nine_trs = randint(0, 1)
        t1_distinct = randint(5, 600)
        t2_distinct = np.floor(np.random.normal(300, 20))
        t3_distinct = np.floor(np.random.normal(150, 20))
        t4_distinct = np.floor(np.random.normal(150, 20))
        is_corr_nod = randint(0, 1)
        amount = randint(1, 10)
        inx_roles = randint(0, 10)
        inx_grpid = randint(0, 10)
        yichu = randint(0, 1)
        fstflg = randint(0, 1)
        value[0] = match_rate
        value[1] = is_nine_trs
        value[2] = t1_distinct
        value[3] = t2_distinct
        value[4] = t3_distinct
        value[5] = t4_distinct
        value[6] = is_corr_nod
        value[-6] = amount
        value[-4] = inx_roles
        value[-3] = inx_grpid
        value[-2] = yichu
        value[-1] = fstflg
        # 任务id
        wkiid = str(i) + str(j)
        data_dict[wkiid] = value
    print('data_dict: %s' % data_dict)
    # 打分
    # mlscore = MLWkiidScore()
    score_dict = get_wikiid_score(data_dict)
    for key, val in data_dict.items():
        score = score_dict.get(key, 0)
        if score < 0 or score > 1:
            print('score err:' + str(score))
            continue
        y = round((1.1 - score) * 600, 2)
        oneData = [key]
        oneData.extend(val[:7])
        oneData.append(val[-6])
        oneData.extend(val[-4:])
        oneData.append(y)
        print('oneData:%s' % oneData)
        sql = sqlPre % tuple(oneData)
        # sql = sql % (wkiid,match_rate, is_nine_trs, t1_distinct, t2_distinct, t3_distinct, t4_distinct, is_corr_nod,
        #                   amount,inx_roles, inx_grpid, yichu, fstflg,y)
        insertCur = conn.cursor()
        insertCur.execute(sql)
    conn.commit()


def split_data(df, labels):
    '''划分数据集'''
    return train_test_split(df.values, labels, test_size=0.3, random_state=2021)

def multi_train_by_sk(df,labels):
    '''
    sk-多元线性回归
    :param df:
    :param labels:
    :return:
    '''
    X_train, X_test, y_train, y_test= split_data(df, labels)
    linReg=linear_model.LinearRegression()
    linReg.fit(X_train,y_train)
    y_hat=linReg.predict(X_test)
    test_loss= computeLoss(y_hat, y_test)
    print('multi_train_by_sk loss: %s' % test_loss)
    print('w: %s, b: %s' % (linReg.coef_,linReg.intercept_))
    return


def computeLoss(y_hat, y_test):
    return np.sum(np.square(y_hat - y_test)) / len(y_test)


def nn_train_by_ks(df,labels):
    '''
    keras-神经网络
    :param df:
    :param labels:
    :return:
    '''
    X_train, X_test, y_train, y_test= split_data(df, labels)
    model=ks.Sequential([
        ks.layers.Dense(20,activation=tf.nn.sigmoid),
        ks.layers.Dropout(0.2),
        ks.layers.Dense(20,activation=tf.nn.sigmoid),
        ks.layers.Dropout(0.2),
        ks.layers.Dense(20,activation=tf.nn.sigmoid),
        ks.layers.Dropout(0.2),
        ks.layers.Dense(1,activation=tf.nn.relu)
    ])

    model.compile(optimizer=tf.train.AdamOptimizer(learning_rate=0.05),loss='mse')
    model.fit(X_train,y_train,epochs=200)
    y_hat=model.predict(X_test)
    loss=computeLoss(y_hat,y_test)
    print('nn_train_by_ks loss: %s' % loss)

    return

def train_xgb_by_sk(df,labels):
    X_train, X_test, y_train, y_test= split_data(df, labels)
    dtrain = xgb.DMatrix(X_train, y_train)
    # dval = xgb.DMatrix(X_test, y_test)
    dtest = xgb.DMatrix(X_test)
    #
    # param = {'booster': 'gbtree',
    #          'gamma': 0.1,
    #          'subsample': 0.8,
    #          'colsample_bytree': 0.8,
    #          'max_depth': 6,
    #          'eta': 0.03,
    #          'silent': 1,
    #          'objective': 'reg:linear',
    #          'nthread': 4,
    #          'eval_metric': 'auc'}
    # watchlist = [(dtrain, 'train'), (dval, 'val')]
    # model = xgb.train(param, dtrain, num_boost_round=100, evals=watchlist)

    model = xgb.XGBRegressor(max_depth=5, learning_rate=0.1, n_estimators=120, silent=False, objective='reg:gamma')
    model.fit(X_train, y_train)
    y_hat=model.predict(X_test)
    loss=computeLoss(y_hat,y_test)
    print('train_xgb_by_sk loss %s' % loss)




def normalize0(train_x):
    return (train_x-train_x.min(axis=0))/(train_x.max(axis=0)-train_x.min(axis=0))

if __name__ == '__main__':
    print('main start')
    # gen_test_data()
    conn=getConn()
    sql="SELECT match_rate, is_nine_trs, t1_distinct, t2_distinct, t3_distinct, t4_distinct,"\
    "is_corr_nod,amount,inx_roles, inx_grpid, yichu, fstflg,y"\
    " from his_task_used_time "\
    "ORDER BY id "\
    "limit 10000"
    df=pd.read_sql(sql,conn)
    labels=df['y'].values
    del df['y']
    df=normalize0(df)
    multi_train_by_sk(df,labels)
    nn_train_by_ks(df,labels)
    train_xgb_by_sk(df,labels)
    conn.close()

    print('main done')
