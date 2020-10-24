import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

#加载数据集
boston_housing = tf.keras.datasets.boston_housing
(train_x,train_y),(test_x,test_y) = boston_housing.load_data()

num_train=len(train_x)   #训练集和测试机中样本的数量
num_test=len(test_x)

#对训练样本和测试样本进行标准化（归一化）,这里有用到张量的广播运算机制
x_train=(train_x-train_x.min(axis=0))/(train_x.max(axis=0)-train_x.min(axis=0))
y_train = train_y

x_test=(test_x-test_x.min(axis=0))/(test_x.max(axis=0)-test_x.min(axis=0))
y_test = test_y

#生成多元回归需要的二维形式
x0_train = np.ones(num_train).reshape(-1,1)
x0_test = np.ones(num_test).reshape(-1,1)

#对张量数据类型转换和进行堆叠
X_train = tf.cast(tf.concat([x0_train,x_train],axis=1), tf.float32)
X_test = tf.cast(tf.concat([x0_test, x_test], axis=1), tf.float32)

#将房价转换为列向量
Y_train = tf.constant(y_train.reshape(-1,1), tf.float32)
Y_test = tf.constant(y_test.reshape(-1,1), tf.float32)

#设置超参数
learn_rate = 0.01
iter = 2000
display_step=200

#设置模型变量初始值
np.random.seed(612)
W = tf.Variable(np.random.randn(14,1), dtype = tf.float32)

#训练模型
mse_train=[]
mse_test=[]

for i in range(iter+1):
    with tf.GradientTape() as tape:
        PRED_train = tf.matmul(X_train,W)
        Loss_train = 0.5*tf.reduce_mean(tf.square(Y_train-PRED_train))

        PRED_test = tf.matmul(X_test,W)
        Loss_test = 0.5*tf.reduce_mean(tf.square(Y_test-PRED_test))

    mse_train.append(Loss_train)
    mse_test.append(Loss_test)

    dL_dW = tape.gradient(Loss_train, W)
    W.assign_sub(learn_rate*dL_dW)

    if i % display_step == 0:
        print('i: %i, Train_loss:%f, Test_loss: %f' % (i,Loss_train,Loss_test))


#可视化输出
plt.figure(figsize=(20,10))

plt.subplot(221)
plt.ylabel('MSE')
plt.plot(mse_train,color = 'blue',linewidth=3)
plt.plot(mse_test,color = 'red',linewidth=3)
plt.title('训练误差和测试误差',fontsize = 20)

plt.subplot(222)
plt.ylabel('Price')
plt.plot(y_train,color='blue', marker='o', label='true_price')
plt.plot(PRED_train, color ='red', marker='.', label='predict')
plt.legend()
plt.title('训练数据集房价和训练数据集预测房价',fontsize = 20)

plt.subplot(223)
plt.ylabel('Price')
plt.plot(y_test, color='blue', marker='o', label='true_price')
plt.plot(PRED_test, color='red', marker='.', label='predict')
plt.legend()
plt.title('测试数据集房价和测试数据集预测房价',fontsize = 20)

plt.show()