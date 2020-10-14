import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

'''波士顿房价预测：一元和多元线性回归'''

def normalize0(train_x):
    return (train_x-train_x.min(axis=0))/(train_x.max(axis=0)-train_x.min(axis=0))

def normalize(X):
    mean=np.mean(X)
    std=np.std(X)
    X=(X-mean)/std
    return X

def append_bias_reshape(feats,labels):
    # 样本数
    m=feats.shape[0]
    # 特征数
    n=feats.shape[1]
#     np.c_：是按行连接两个矩阵，就是把两矩阵左右相加，要求行数相等；按行相加，相当于增添特征
# np.r_：是按列连接两个矩阵，就是把两矩阵上下相加，要求列数相等。
    x=np.reshape(np.c_[np.ones(m),feats],[m,n+1])
    y=np.reshape(labels,[m,1])
    return x,y

(x_train, y_train), (x_test, y_test)=tf.keras.datasets.boston_housing.load_data()


def train_multi():
    '''
    多元线性回归
    :return:
    '''
    global x_train
    print(x_train.shape)
    print(y_train.shape)
    # 归一化 防止loss出现nan
    x_train = normalize0(x_train)
    # 变换X维度，使得wx+b变为 wx
    x, y = append_bias_reshape(x_train, y_train)
    print(x.shape)
    print(y.shape)
    m = len(x)
    n = 13 + 1
    X = tf.placeholder(tf.float32, shape=[1, n], name='X')
    Y = tf.placeholder(tf.float32, name='Y', shape=[1, 1])
    w = tf.Variable(tf.random_normal([n, 1]))
    # model
    y_hat = tf.matmul(X, w)
    # 损失
    loss = tf.square(Y - y_hat, name='loss')
    # 梯度优化器
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)
    # 全局初始化Variable
    init_op = tf.global_variables_initializer()
    # 每次迭代的损失数组
    epoch_loss_arr = []
    with tf.Session() as sess:
        sess.run(init_op)
        print('init op done')
        writer = tf.summary.FileWriter(logdir='graphs', graph=sess.graph)
        for i in range(100):
            total_loss = 0.0
            for f, label in zip(x, y):
                lo, op = sess.run([loss, optimizer],
                                  feed_dict={X: np.reshape(f, [1, len(f)]), Y: np.reshape(label, [1, 1])})
                total_loss += lo[0][0]
            print('epoch {0},loss {1}'.format(i, total_loss / m))
            epoch_loss_arr.append(total_loss / m)

        writer.close()
        w_val = sess.run(w)
    # 绘制损失图
    plt.plot(epoch_loss_arr)
    plt.show()
    # plt.savefig('multi_loss.png')
    N = 400
    x_new = x[N, :]
    # round(1)四舍五入 保留1位小数
    y_pred = np.matmul(x_new, w_val).round(1)
    print(y_pred)
    print(y_pred[0])
    print(y[N])


# train_multi()


def train_one():
    '''
    一元线性回归
    :return:
    '''
    global x_train, x_test
    # 只取X中下标为5的特征
    x_train = x_train[:, 5]
    x_test = x_test[:, 5]
    print('x_train:', x_train[:5])
    print('y_train:', y_train[:5])
    n_samples = len(x_train)
    print('n_samples:', n_samples)
    X = tf.placeholder(tf.float32, name='X')
    Y = tf.placeholder(tf.float32, name='Y')
    # 声明 w b
    b = tf.Variable(0.0)
    w = tf.Variable(0.0)
    # 模型
    Y_hat = X * w + b
    loss = tf.square(Y - Y_hat, name='loss')
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)
    init_op = tf.global_variables_initializer()
    total = []
    with tf.Session() as sess:
        sess.run(init_op)
        writer = tf.summary.FileWriter('graphs', sess.graph)
        # 迭代100次
        for i in range(100):
            total_loss = 0
            for x, y in zip(x_train, y_train):
                _, l = sess.run([optimizer, loss], feed_dict={X: x, Y: y})
                print('optimizer,loss:', _, l)
                total_loss += l
            total.append(total_loss / n_samples)
            print('epoch {0}: Loss {1}'.format(i, total_loss / n_samples))
        writer.close()
        b_val, w_val = sess.run([b, w])

    # 计算预测值
    y_pred = x_train * w_val + b_val
    print('done')
    # 绘制散点图和预测曲线
    plt.plot(x_train, y_train, 'bo', label='real_y')
    plt.plot(x_train, y_pred, 'r', label='pred_y')
    # 设置图例
    plt.legend()
    plt.show()
    # 绘制损失值和迭代次数曲线
    plt.plot(total)
    # 保存图片
    plt.savefig('xx.jpg')
    plt.show()


# train_one()

# from sklearn.datasets import load_boston
# batch_size=10
# number_feats=14
# boston=load_boston()

# def data_gen(filename):
#     f_queue=tf.train.string_input_producer(filename)
#     reader=tf.TextLineReader(skip_header_lines=1)
#     _,value=reader.read(f_queue)
#     record_defaults=[[0.0] for _ in range(number_feats)]
#     data=boston
#     feats=tf.stack(tf.gather_nd(data,[[5],[10],[12]]))
#     label=data[-1]
#     dad=10*batch_size
#     cap=20*batch_size
#     feat_batch,label_batch=tf.train.shuffle_batch([feats,label],batch_size=batch_size,min_after_dequeue=dad,capacity=cap)
#
#     return feat_batch,label_batch
#
# def gen_data(feat_batch,label_batch):
#     with tf.Session() as sess:
#         coord=tf.train.Coordinator()
#         threads=tf.train.start_queue_runners(coord=coord)
#         for _ in range(5):
#             feats,labels=sess.run([feat_batch,label_batch])
#             print(feats,"HI")
#             coord.request_stop()
#             coord.join(threads)
#
#
# if __name__ == '__main__':
#     feat_batch,label_batch=data_gen([data_file])
#     gen_data(feat_batch,label_batch)