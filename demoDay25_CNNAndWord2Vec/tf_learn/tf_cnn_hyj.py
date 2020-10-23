'''
CNN练习
data: mnist
'''
from __future__ import division, print_function
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('G:\\bigdata\\badou\\00-data\\MNIST2', one_hot=True)


def train_size(num):
    x_train = mnist.train.images[:num]
    y_train = mnist.train.labels[:num]
    print('x:%s, y:%s' % (x_train.shape, y_train.shape))
    return x_train, y_train


def test_size(num):
    x_train = mnist.test.images[:num]
    y_train = mnist.test.labels[:num]
    print('x:%s, y:%s' % (x_train.shape, y_train.shape))
    return x_train, y_train


def display_digit(num, x_train, y_train):
    print(y_train[num])
    # 对于一维数组 默认是列向量，axis=0表示每列比较 取最大值 axis=1表示每行比较取最大值
    label = y_train[num].argmax(axis=0)
    image = x_train[num].reshape([28, 28])
    plt.title('example %s, label %s' % (num, label))
    plt.imshow(image, cmap=plt.get_cmap('gray_r'))
    plt.show()


def display_mult_flat(start, stop, x_train):
    images = x_train[start].reshape([1, 784])
    for i in range(start + 1, stop):
        images = np.concatenate((images, x_train[i].reshape([1, 784])))
        plt.imshow(images, cmap=plt.get_cmap('gray_r'))
        plt.show()


x_train, y_train = train_size(1000)
# 图片展示label
display_digit(np.random.randint(0, x_train.shape[0]), x_train, y_train)
# 图片展示0-400张图片的像素
display_mult_flat(0, 400, x_train)
# 定义参数
eta = 0.001
max_ite = 500
batch_size = 128
display_step = 10
n_input = 784
n_classes = 10
drop_out = 0.85
x = tf.placeholder(dtype=tf.float32, shape=[None, n_input])
y = tf.placeholder(dtype=tf.float32, shape=[None, n_classes])
keep_prob = tf.placeholder(tf.float32)


# 定义一个输入为 x，权值为 W（），偏置为 b，给定步幅的卷积层。激活函数是 ReLU，padding 设定为 SAME 模式
def conv2d(x, W, b, strides=1):
    x = tf.nn.conv2d(x, filter=W, strides=[1, strides, strides, 1], padding='SAME')
    return tf.nn.relu(tf.nn.bias_add(x, b))


# 定义一个输入是 x 的 maxpool 层，卷积核为 ksize 并且 padding 为 SAME
def maxpool2d(x, k=2):
    # ksize：池化窗口的大小，取一个四维向量，一般是[1, height, width, 1]，因为我们不想在batch和channels上做池化，所以这两个维度设为了1
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')


# 定义网络层的权重和偏置。第一个 conv 层有一个 5×5 的卷积核，1 个输入和 32 个输出。
# 第二个 conv 层有一个 5×5 的卷积核，32 个输入和 64 个输出。
# 全连接层有 7×7×64 个输入和 1024 个输出，而第二层有 1024 个输入和 10 个输出对应于最后的数字数目。
# 所有的权重和偏置用 randon_normal 分布完成初始化
weights = {
    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
    'wd1': tf.Variable(tf.random_normal([7 * 7 * 64, 1024])),
    'out': tf.Variable(tf.random_normal([1024, n_classes]))
}
biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([n_classes])),
}


# 定义 convnet，其构成是两个卷积层，然后是全连接层，一个 dropout 层，最后是输出层
def conv_net(x, weights, biases, dropout):
    # -1表示缺省值 其他shape维度确定后，-1表示总数除以你们几个的乘积，该是几就是几
    x = tf.reshape(x, [-1, 28, 28, 1])
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    conv1_maxpool = maxpool2d(conv1)
    conv2 = conv2d(conv1_maxpool, weights['wc2'], biases['bc2'])
    conv2_maxpool = maxpool2d(conv2)
    fc1 = tf.reshape(conv2_maxpool, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.nn.relu(
        tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    )
    fc1 = tf.nn.dropout(fc1, keep_prob=dropout)
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out


# 建立一个给定权重和偏置的 convnet。定义基于 cross_entropy_with_logits 的损失函数，并使用 Adam 优化器进行损失最小化。
# 优化后，计算精度
pred = conv_net(x, weights, biases, keep_prob)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred))
optimizer = tf.train.AdamOptimizer(learning_rate=eta).minimize(loss)
correct_bool_arr = tf.equal(tf.argmax(y, axis=1), tf.argmax(pred, axis=1))
auc = tf.reduce_mean(tf.cast(correct_bool_arr, tf.float32))
init = tf.global_variables_initializer()

train_loss = []
test_auc = []
train_auc = []

with tf.Session() as sess:
    sess.run(init)
    step = 1
    while step <= max_ite:
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, keep_prob: drop_out})
        if step % display_step == 0:
            loss_val, auc_val = sess.run([loss, auc], feed_dict={x: batch_x, y: batch_y, keep_prob: drop_out})
            test_loss_val, test_auc_val = sess.run([loss, auc], feed_dict={x: mnist.test.images,
                                                                           y: mnist.test.labels, keep_prob: drop_out})
            print('step %s,train_loss %s, test_loss %s,train_auc %s, test_auc %s'
                  % (step, loss_val, test_loss_val, auc_val, test_auc_val)
                  )

            train_loss.append(loss_val)
            train_auc.append(auc_val)
            test_auc.append(test_auc_val)
        step += 1

# 画出每次迭代的 Softmax 损失以及训练和测试的精度
eval_indices=range(0,max_ite,display_step)
plt.plot(eval_indices,train_loss,'k-')
plt.title('softmax loss per iteration')
plt.xlabel('iteration')
plt.ylabel('softmax loss')
plt.show()


plt.plot(eval_indices,train_auc,'k-',label='train auc')
plt.plot(eval_indices,test_auc,'r--',label='test auc')
plt.title('train and test auc')
plt.xlabel('iteration')
plt.ylabel('auc')
plt.legend(loc='lower right')
plt.show()

