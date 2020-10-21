'''
Contrib 可以用来添加各种层到神经网络模型，如添加构建块。这里使用的一个方法是 tf.contrib.layers.fully_connected
'''
import tensorflow as tf
import tensorflow.contrib.layers as layers
from tensorflow.python import debug as tf_debug
from tensorflow.examples.tutorials.mnist import input_data

# 1 定义网络参数
n_input=784
n_hidden=30
n_classes=10
# 2 超参数
batch_size=200
eta=0.001
max_epoch=10
# 3读取数据
mnist = input_data.read_data_sets('G:\\bigdata\\badou\\00-data\\MNIST2', one_hot=True)
'''
这是一个分类问题，所以最好使用交叉熵损失，隐藏层使用 ReLU 激活函数，输出层使用 softmax 函数
'''
def multilayer(x):
    fc1=layers.fully_connected(x,n_hidden,activation_fn=tf.nn.relu,scope='fc1')
    fc2=layers.fully_connected(fc1,256,activation_fn=tf.nn.relu,scope='fc2')
    out=layers.fully_connected(fc1,n_classes,activation_fn=None,scope='out')
    return out

x=tf.placeholder(dtype=tf.float32,shape=[None,n_input])
y=tf.placeholder(dtype=tf.float32,shape=[None,n_classes])
y_hat=multilayer(x)
# softmax+交叉熵
loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=y_hat))
# mse损失
# loss=tf.reduce_mean(tf.square(y-y_hat))
# 优化器
optimizer=tf.train.AdamOptimizer(learning_rate=eta).minimize(loss)
# optimizer=tf.train.GradientDescentOptimizer(learning_rate=eta).minimize(loss)
# 变量初始化
init=tf.global_variables_initializer()
# auc
correct_pred=tf.equal(tf.argmax(y,axis=1),tf.argmax(y_hat,axis=1))
auc=tf.reduce_mean(tf.cast(correct_pred,tf.float32))

with tf.Session() as sess:
    sess.run(init)
    # 迭代次数
    for epoch in range(max_epoch):
        # 每次迭代的步数
        steps=int(mnist.train.num_examples/batch_size)
        for i in range(steps):
            batch_x,batch_y=mnist.train.next_batch(batch_size)
            _=sess.run([optimizer],feed_dict={x:batch_x,y:batch_y})
            # print('epoch %s,batch %s, loss %s'%(epoch,i,l_val))

        # loss_val,auc_val=sess.run([loss,auc],feed_dict={x:mnist.test.images,y:mnist.test.labels})
        # eval方式执行图计算
        auc_val=auc.eval(feed_dict={x:mnist.test.images,y:mnist.test.labels})
        loss_val=loss.eval(feed_dict={x:mnist.test.images,y:mnist.test.labels})
        print('epoch %s,loss %s, auc %s'%(epoch,loss_val,auc_val))
        # 获取第一个隐藏层
        fc1=tf.get_default_graph().get_tensor_by_name('fc1/weights:0')
        # 输出fc1的权重
        print(sess.run(fc1,feed_dict={x:mnist.test.images,y:mnist.test.labels}))

#         epoch 9,loss 0.145464, auc 0.9574
