# 1 导入模块
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
# 2 读取数据
mnist=input_data.read_data_sets("F:\\00-data\\MNIST",one_hot=True)
'''
3
定义超参数和其他常量。这里，每个手写数字的尺寸是 28×28=784 像素。数据集被分为 10 类，以 0 到 9 之间的数字表示。
这两点是固定的。学习率、最大迭代周期数、每次批量训练的批量大小以及隐藏层中的神经元数量都是超参数。
可以通过调整这些超参数，看看它们是如何影响网络表现的
'''
n_input=784
n_classes=10
max_epoches=10000
learning_rate=0.1
batch_size=50
seed=0
n_hidden=30

#4 Sigmoid 函数的导数
def sigmaprime(x):
    return tf.multiply(tf.sigmoid(x),tf.subtract(tf.constant(1.0),tf.sigmoid(x)))

#5 为训练数据创建占位符
x_in=tf.placeholder(tf.float32,[None,n_input])
y=tf.placeholder(tf.float32,[None,n_classes])

# 6 创建模型
def multilayer(x,w,b):
    # 隐藏层net值
    h_1=tf.add(tf.matmul(x,w['h1']),b['h1'])
    # 隐藏层out值
    h1_out=tf.sigmoid(h_1)
    # 输出层net值
    res_net=tf.add(tf.matmul(h1_out,w['res']),b['res'])
    return tf.sigmoid(res_net),res_net,h1_out,h_1

# 7定义权重和偏置变量
w={'h1':tf.Variable(tf.random_normal([n_input,n_hidden],seed=seed)),
   'res':tf.Variable(tf.random_normal([n_hidden,n_classes],seed=seed))
   }

b={'h1':tf.Variable(tf.random_normal([1,n_hidden],seed=seed)),
   'res':tf.Variable(tf.random_normal([1,n_classes],seed=seed))
   }

# 8为正向传播、误差、梯度和更新计算创建计算图
y_hat,y_net,h_out,h_net=multilayer(x_in,w,b)
err=y_hat-y
# bp
# 同位置元素相乘
delta_2=tf.multiply(err,sigmaprime(y_net))
# h_out=tf.reshape(h_out,shape=[1,30])
# delta_2=tf.reshape(delta_2,shape=[1,10])
# 矩阵相乘
delta_w_2=tf.matmul(tf.transpose(h_out),delta_2)
print(h_out,delta_2,delta_w_2)

wtd_err=tf.matmul(delta_2,tf.transpose(w['res']))
delta_1=tf.multiply(wtd_err,sigmaprime(h_net))
delta_w_1=tf.matmul(tf.transpose(x_in),delta_1)

eta=tf.constant(learning_rate)

# 更新权重w b
step=[
    # assign表示赋值 即用右边的值更新w['h1']的值
    tf.assign(w['h1'],tf.subtract(w['h1'],tf.multiply(eta,delta_w_1))),
    # reduce_mean(delta_1,axis=0)  axis=0表示相同列求平均
    tf.assign(b['h1'],tf.subtract(b['h1'],tf.multiply(eta,tf.reduce_mean(delta_1,axis=0)))),
    tf.assign(w['res'],tf.subtract(w['res'],tf.multiply(eta,delta_w_2))),
    tf.assign(b['res'],tf.subtract(b['res'],tf.multiply(eta,tf.reduce_mean(delta_2,axis=0)))),
]

# 9定义计算精度 accuracy 的操作
auc_bool_arr=tf.equal(tf.argmax(y,1),tf.argmax(y_hat,1))
auc=tf.reduce_mean(tf.cast(auc_bool_arr,tf.float32))

# 10初始化变量
init=tf.global_variables_initializer()
print('train num:%s, test num: %s'%(len(mnist.train.images),len(mnist.test.images)))

# 11 执行图
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(max_epoches):
        batch_xs,batch_ys=mnist.train.next_batch(batch_size)
        sess.run(step,feed_dict={x_in:batch_xs,y:batch_ys})
        if epoch%10==0:
            auc_train=sess.run(auc,feed_dict={x_in:mnist.train.images,y:mnist.train.labels})
            auc_test=sess.run(auc,feed_dict={x_in:mnist.test.images,y:mnist.test.labels})
            print('epoch %s, auc_train %s, auc_test %s'%(epoch,auc_train,auc_test))

