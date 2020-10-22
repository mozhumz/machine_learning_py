import tensorflow as tf
import tensorflow.contrib.layers as layers
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,Normalizer
import pandas as pd
import seaborn as sns

# 加载数据集并创建 Pandas 数据帧来分析数据
boston=datasets.load_boston()
df=pd.DataFrame(boston.data,columns=boston.feature_names)
df['target']=boston.target
df.describe()

_,ax=plt.subplots(figsize=(12,10))
# 相关性分析
corr=df.corr(method='pearson')
# 发散调色板
cmap=sns.diverging_palette(220,10,as_cmap=True)
sns.heatmap(corr,square=True,cmap=cmap,cbar_kws={'shrink':.9},ax=ax,annot=True,annot_kws={'fontsize':12})
plt.show()
# 可以看到三个参数 RM、PTRATIO 和 LSTAT 在幅度上与输出之间具有大于 0.5 的相关性。选择它们进行训练
X_train, X_test, y_train, y_test=train_test_split(df[['RM','PTRATIO','LSTAT']],df[['target']],test_size=0.3,random_state=0)
# 归一化
X_train=MinMaxScaler().fit_transform(X_train)
X_test=MinMaxScaler().fit_transform(X_test)
y_train=MinMaxScaler().fit_transform(y_train)
y_test=MinMaxScaler().fit_transform(y_test)

#定义常量和超参数
m=len(X_train)
n_input=3
n_hidden=20
batch_size=200
eta=0.01
max_epoch=2000

def multiplayer(x):
    fc1=layers.fully_connected(x,num_outputs=n_hidden,activation_fn=tf.nn.relu,scope='fc1')
    out=layers.fully_connected(fc1,num_outputs=1,activation_fn=tf.nn.sigmoid,scope='out')
    return out

x=tf.placeholder(dtype=tf.float32,shape=[None,n_input],name='X')
y=tf.placeholder(dtype=tf.float32,shape=[None,1],name='y')
y_hat=multiplayer(x)
loss=tf.reduce_mean(tf.square(y-y_hat))
optimizer=tf.train.AdamOptimizer(learning_rate=eta).minimize(loss)
init=tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    writer=tf.summary.FileWriter('graphs',sess.graph)
    for i in range(max_epoch):

        _,l,p=sess.run([optimizer,loss,y_hat],feed_dict={x:X_train,y:y_train})

        if i%100==0:
            print('epoch %s, loss %s'%(i,l))

    print('train done')
    # 测试集mse
    correct_pred=tf.square(y-y_hat)
    auc=tf.reduce_mean(tf.cast(correct_pred,tf.float32))
    mse_val=auc.eval({x:X_test,y:y_test})
    print('mse %s'%(mse_val))
    # 测试集真实值和估计值散点图
    y_hat_val=y_hat.eval({x:X_test,y:y_test})
    plt.scatter(y_test,y_hat_val)
    plt.show()
    writer.close()



