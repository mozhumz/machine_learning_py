'''tensorflow逻辑回归LR实现'''
import tensorflow as tf
from matplotlib import pyplot as plt,image as mpimg
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('F:\\八斗学院\视频\\14期正式课\\00-data\\MNIST', one_hot=True)

x=tf.placeholder(tf.float32,shape=[None,784],name='X')
y=tf.placeholder(tf.float32,shape=[None,10],name='Y')
W=tf.Variable(tf.zeros([784,10],name='W'))
b=tf.Variable(tf.zeros([10]),name='b')

with tf.name_scope('wx_b') as scope:
    y_hat=tf.nn.softmax(tf.matmul(x,W)+b)

# 训练时添加 summary 操作来收集数据。使用直方图以便看到权重和偏置随时间相对于彼此值的变化关系。可以通过 TensorBoard Histogtam 选项卡看到
w_h=tf.summary.histogram('weights',W)
b_h=tf.summary.histogram('bias',b)

# 定义交叉熵（cross-entropy）和损失（loss）函数，并添加 name scope 和 summary 以实现更好的可视化
with tf.name_scope('cross-entropy') as scope:
    loss=tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=y_hat)
    # 使用 scalar summary 来获得随时间变化的损失函数。scalar summary 在 Events 选项卡下可见
    tf.summary.histogram('cross-entropy',loss)

# 采用 TensorFlow GradientDescentOptimizer，学习率为 0.01。为了更好地可视化，定义一个 name_scope
with tf.name_scope('train') as scope:
    optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)


# 变量初始化器
init=tf.global_variables_initializer()
# 组合所有的 summary 操作
merge_summary=tf.summary.merge_all()
batch_size=50
correct_pred = tf.equal(tf.argmax(y_hat, 1), tf.argmax(y, 1))  # [0,0,1,1,1]
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
with tf.Session() as sess:
    sess.run(init)
    writer=tf.summary.FileWriter('graphs',graph=sess.graph)
    for epoch in range(2000):
        loss_avg=0.
        num_of_batch=int(mnist.train.num_examples/batch_size)
        for i in range(num_of_batch):
            batch_xs,batch_ys=mnist.train.next_batch(50)
            _,l,_summary_str=sess.run([optimizer,loss,merge_summary],feed_dict={x:batch_xs,y:batch_ys})
            loss_avg+=l
            writer.add_summary(_summary_str,epoch*num_of_batch+i)
        loss_avg/=num_of_batch
        print('epoch {0},loss {1}'.format(epoch,loss_avg))
    print('done')
    print(sess.run([accuracy],feed_dict={x:mnist.test.images,y:mnist.test.labels}))
