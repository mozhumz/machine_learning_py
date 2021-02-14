from sklearn import datasets
import matplotlib.pyplot as plt

from sklearn.model_selection import learning_curve,validation_curve
from sklearn.svm import SVC
from sklearn.datasets import load_digits
import numpy as np
import pickle
import joblib

def test_make_regression():
    '''创建回归数据'''
    X,y=datasets.make_regression(n_samples=100,n_features=1,n_targets=1,noise=2)
    plt.scatter(X,y)
    plt.show()
    from sklearn import metrics
    print(sorted(metrics.SCORERS.keys()))
    return

def test_over_fitting1():
    '''
    过拟合模拟-learning_curve
    gamma=0.01 train_size=0.25 交叉验证最优
    gamma=0.001，交叉验证损失和train_size成反比
    :return:
    '''
    (X,y)=load_digits(return_X_y=True)
    # train_sizes表示划分数据集时，训练集的比重
    train_sizes,train_loss,test_loss=learning_curve(SVC(gamma=0.01),X,y,cv=10,
                                                    train_sizes=[0.1,0.25,0.5,0.75,1],
                                                    scoring='neg_mean_squared_error')
    train_loss=-np.mean(train_loss,axis=1)
    test_loss=-np.mean(test_loss,axis=1)

    plt.plot(train_sizes,train_loss,'o-',color='r',label='train')
    plt.plot(train_sizes,test_loss,'o-',color='g',label='cross')
    plt.xlabel('train_size')
    plt.ylabel('loss')
    # 右上角显示图例
    plt.legend(loc='best')
    plt.show()

    return

def test_over_fitting2():
    '''
    过拟合模拟-validation_curve
    gamma值范围[0.0005,0.002] 共10个点，在第7个点时交叉验证最优，第8个点开始过拟合
    :return:
    '''
    (X,y)=load_digits(return_X_y=True)
    gamma_range=np.linspace(0.0005,0.002,10)
    # train_sizes表示划分数据集时，训练集的比重
    train_loss,test_loss=validation_curve(SVC(),X,y,param_name='gamma',param_range=gamma_range,cv=10,

                                                    scoring='neg_mean_squared_error')
    # axis=1按行求均值
    train_loss=-np.mean(train_loss,axis=1)
    test_loss=-np.mean(test_loss,axis=1)

    plt.plot(gamma_range,train_loss,'o-',color='r',label='train')
    plt.plot(gamma_range,test_loss,'o-',color='g',label='test')
    plt.xlabel('gamma')
    plt.ylabel('loss')
    # 右上角显示图例
    plt.legend(loc='best')
    plt.show()
    return

def test_save_restore_model():
    '''
    保存和加载模型
    :return:
    '''
    (X,y)=datasets.load_iris(return_X_y=True)
    model=SVC()
    model.fit(X,y)
    # # 保存 model.joblib后缀可以任意
    with open('./tmp/model.joblib','wb') as f:
        # pickle.dump(model,f)
        joblib.dump(model,f)
    # 加载
    with open('./tmp/model.joblib','rb') as f:
        # model2=pickle.load(f)
        model2=joblib.load(f)
        print(model2.predict(X[:2]))


    return

def test_file():
    import os
    import pathlib
    cu=os.getcwd()
    print(cu)
    f1=os.path.join(cu,'tmp/model/d1/d2/t.txt')
    flag1=os.path.isdir(f1)
    flag2=os.path.isfile(f1)
    print(flag1,flag2)
    print(os.path.abspath(os.path.dirname(f1)))
    pathlib.Path(f1).touch()
    return

if __name__ == '__main__':
    print('main start')
    # test_over_fitting2()
    # test_save_restore_model()
    test_file()
    print('main end')