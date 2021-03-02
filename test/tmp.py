params = {'num_leaves': 38,
          'min_data_in_leaf': 50,
          'objective': 'regression',
          'max_depth': -1,
          'learning_rate': 0.02,
          "min_sum_hessian_in_leaf": 6,
          "boosting": "gbdt",
          "feature_fraction": 0.9,
          "bagging_freq": 1,
          "bagging_fraction": 0.7,
          "bagging_seed": 11,
          "lambda_l1": 0.1,
          "verbosity": -1,
          "nthread": 4,
          'metric': 'mae',#absolute loss 绝对值损失,即 l1
          "random_state": 2019,
          # 'device': 'gpu'
          }

params = {'num_leaves': 60, #结果对最终效果影响较大，越大值越好，太大会出现过拟合
          'min_data_in_leaf': 30,
          'objective': 'binary', #定义的目标函数
          'max_depth': -1,
          'learning_rate': 0.03,
          "min_sum_hessian_in_leaf": 6,
          "boosting": "gbdt",
          "feature_fraction": 0.9,	#提取的特征比率
          "bagging_freq": 1,
          "bagging_fraction": 0.8,
          "bagging_seed": 11,
          "lambda_l1": 0.1,				#l1正则
          # 'lambda_l2': 0.001,		#l2正则
          "verbosity": -1,
          "nthread": -1,				#线程数量，-1表示全部线程，线程越多，运行的速度越快
          'metric': {'binary_logloss', 'auc'},	##评价函数选择
          "random_state": 2019,	#随机数种子，可以防止每次运行的结果不一致
          # 'device': 'gpu' ##如果安装的事gpu版本的lightgbm,可以加快运算
          }
import pandas as pd
from fbprophet import Prophet
# 读入数据集
df = pd.read_csv('F:/tmp/pro.csv')
m = Prophet()
m.fit(df)

# 构建待预测日期数据框，periods = 365 代表除历史数据的日期外再往后推 365 天
future = m.make_future_dataframe(periods=365)
forecast = m.predict(future)
print(forecast)
print('---------------')
import pickle,joblib
joblib.dump(m,'pro.joblib')
del m
m=joblib.load('pro.joblib')
future = m.make_future_dataframe(periods=365)
forecast = m.predict(future)
with open('pro.pkl', 'wb') as f:
    # , protocol=pickle.HIGHEST_PROTOCOL
    pickle.dump(m, f)
print('ok')