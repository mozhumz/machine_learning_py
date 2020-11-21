import xgboost as xgb
import pandas as pd
import numpy as np
import lightgbm as lgb
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.externals import joblib
from xgboost import plot_importance,plot_tree
import matplotlib
import matplotlib.pyplot as plt
import os
os.environ["PATH"] += os.pathsep + 'D:\\Program Files\\Graphviz 2.44.1\\bin'
# pip install xgboost,lightgbm,sklearn  anaconda3
IDIR = 'G://bigdata//badou//00-data//'
df_train = pd.read_csv(IDIR + 'train_feat.csv').fillna(0.).to_sparse()
labels = np.load(IDIR + 'labels.npy')

X_train, X_test, y_train, y_test = train_test_split(df_train, labels,
                                                    test_size=0.2,
                                                    random_state=2020)
del df_train
del labels
#
# # ########################### XGB ##################
#
dtrain = xgb.DMatrix(X_train, y_train)
dval = xgb.DMatrix(X_test, y_test)

param = {'booster': 'gbtree',
         'gamma': 0.1,
         'subsample': 0.8,
         'colsample_bytree': 0.8,
         'max_depth': 6,
         'eta': 0.03,
         'silent': 1,
         'objective': 'binary:logistic',
         'nthread': 4,
         'eval_metric': 'auc'}
# watchlist = [(dtrain, 'train'), (dval, 'val')]
# pip install xgboost
model=joblib.load(IDIR+'out/xgb.dat')
# plot_tree(model,num_trees=0)
# plt.show()
#
# plot_importance(model)
# plt.show()

# 对测试集进行预测
dtest = xgb.DMatrix(X_test)
ans = model.predict(dtest)
auc=roc_auc_score(y_test,ans)
print('auc %s'%auc)

model2=joblib.load(IDIR+'out/lgb.dat')
val_data = lgb.Dataset(X_test, y_test)
ans2=model2.predict(val_data.data)
auc2=roc_auc_score(y_test,ans2)
print('auc2 %s'%auc2)