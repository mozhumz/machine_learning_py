'''tensorflow逻辑回归LR实现'''
import tensorflow as tf
from matplotlib import pyplot as plt,image as mpimg
from tensorflow.examples.tutorials.mnist import input_data
import math
mnist = input_data.read_data_sets('F:\\八斗学院\\视频\\14期正式课\\00-data\\MNIST', one_hot=True)

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
    # loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=y_hat))
    # loss2 =tf.reduce_mean(-tf.reduce_sum(y * tf.log(y_hat)))
    # 加入clip_by_value 防止y_hat取对数为0
    loss2 = tf.reduce_mean(-tf.reduce_sum(y*tf.log(tf.clip_by_value(y_hat,1e-10,1.0))))
    # 使用 scalar summary 来获得随时间变化的损失函数。scalar summary 在 Events 选项卡下可见
    # tf.summary.scalar('cross-entropy',loss)
    tf.summary.histogram('cross-entropy',loss2)


# 采用 TensorFlow GradientDescentOptimizer，学习率为 0.01。为了更好地可视化，定义一个 name_scope
with tf.name_scope('train') as scope:
    # optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)
    # optimizer=tf.train.AdagradDAOptimizer(learning_rate=0.01,global_step=tf.cast(200,dtype=tf.int64)).minimize(loss)
    optimizer=tf.train.AdamOptimizer().minimize(loss2)



# 变量初始化器
init=tf.global_variables_initializer()
# 组合所有的 summary 操作
merge_summary=tf.summary.merge_all()
batch_size=50

'''
argmax: 
返回tensor中最大分量的索引,axis=1表示按行取最大值 如tensorflow.argmax ( [1,2,3,10,1])  返回 10对应的索引3
equal:
逐个元素判断，a, b 是不是相等
a = [[1,2,3],[4,5,6]]
b = [[1,0,3],[1,5,1]]
tf.equal(a,b)
[[ True False  True]
 [False  True False]]
'''
correct_pred = tf.equal(tf.argmax(y_hat,axis=1), tf.argmax(y, 1))  # [0,0,1,1,1]
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
with tf.Session() as sess:
    sess.run(init)
    writer=tf.summary.FileWriter('graphs',graph=sess.graph)
    for epoch in range(200):
        loss_avg=0.
        loss_avg2=0.
        # 总的批次数
        num_of_batch=int(mnist.train.num_examples/batch_size)
        for i in range(num_of_batch):
            # 每次训练batch_size的样本量
            batch_xs,batch_ys=mnist.train.next_batch(50)
            y_hat_out,_,l2,_summary_str=sess.run([y_hat,optimizer,loss2,merge_summary],feed_dict={x:batch_xs,y:batch_ys})
            writer.add_summary(_summary_str,epoch*num_of_batch+i)
            if  l2 is not None and math.isnan(l2)==False :
                print('l2 %d'%(l2))
                pred_val,auc_val=sess.run([correct_pred,accuracy],feed_dict={x:batch_xs,y:batch_ys})
                print('num_of_batch: %s,auc_val: %s,pred_val: %s'%(i,auc_val,pred_val))
                loss_avg2+=l2
                # loss_avg+=l
                # for sum in l:
            else:
                print('l or l2 is None:',l2)

        loss_avg/=num_of_batch
        loss_avg2/=num_of_batch
        print('epoch %s,loss %s,loss2:%s'%(epoch,loss_avg,loss_avg2))
    print('done')
    correct_pred_out,auc_out=sess.run([correct_pred,accuracy],feed_dict={x:mnist.test.images,y:mnist.test.labels})
    print('pred %s,auc %s'%(correct_pred_out,auc_out))
