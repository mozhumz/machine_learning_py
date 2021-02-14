'''tf1.9基本使用'''
'''
tf.multiply两个矩阵中对应元素各自相乘，要求矩阵shape相似，如(m,n) (m,n)或者(n,m) 
tf.matmul表示点乘，将矩阵a乘以矩阵b，生成a * b

'''
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def simple_use():
    '''tf简单使用：一元一次'''
    # np.random.rand 生成100个[0,1)的均匀分布随机数
    x_data = np.random.rand(100).astype(np.float32)
    y_data = 0.3 * x_data + 0.2
    W = tf.Variable(initial_value=tf.random_uniform(shape=[1], minval=-1., maxval=1.))
    b = tf.Variable(tf.zeros(1))
    y = W * x_data + b
    loss = tf.reduce_mean(tf.square(y - y_data))
    optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
    init = tf.initialize_all_variables()
    with tf.Session() as sess:
        sess.run(init)
        for i in range(201):
            sess.run(optimizer)
            if i % 20 == 0:
                print(i, sess.run(W), sess.run(b))


def test1():
    one = tf.constant(2, shape=[1, 2])
    one2 = tf.constant(3, shape=[2, 1])
    with tf.Session() as sess:
        print(sess.run(tf.multiply(one, one2)))


def test_variable():
    v = tf.Variable(0, 'v1')
    one = tf.constant(1)
    new_v = v + one
    update = tf.assign(v, new_v)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for i in range(3):
            sess.run(update)
            print(sess.run(v))


def add_layer(inputs, in_size, out_size, activa_fn=None):
    outputs = None
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1,dtype=tf.float32)
    outputs = tf.matmul(inputs, Weights) + biases
    if activa_fn:
        outputs = activa_fn(outputs)

    return outputs

def test_nn1():
    '''
    1层神经网络
    :return:
    '''
    # np.newaxis的作用是增加一个维度
    x_data=np.linspace(-1,1,300,dtype=np.float32)[:,np.newaxis]
    print(x_data)
    noise=np.random.normal(0,0.05,x_data.shape)
    y_data=np.square(x_data)-0.5+noise

    xs=tf.placeholder(dtype=tf.float32,shape=[None,1])
    ys=tf.placeholder(dtype=tf.float32,shape=[None,1])
    lay1=add_layer(xs,1,10,activa_fn=tf.nn.relu)
    y_hat=add_layer(lay1,10,1,activa_fn=None)
    loss=tf.reduce_mean((tf.square(y_hat-ys)))
    optimizer=tf.train.AdamOptimizer(0.1).minimize(loss)
    init=tf.global_variables_initializer()
    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    ax.scatter(x_data,y_data)
    # 连续plot
    plt.ion()
    plt.show()

    with tf.Session() as sess:
        sess.run(init)

        for i in range(1000):
            sess.run(optimizer,feed_dict={xs:x_data,ys:y_data})
            if i%10==0:
                print('loss: %s %s' % (i,sess.run(loss,feed_dict={xs:x_data,ys:y_data})))

                # try:
                #     ax.lines.remove(lines[0])
                # except Exception:
                #     pass
                y_hat_val=sess.run(y_hat,feed_dict={xs:x_data})

                lines=ax.plot(x_data,y_hat_val,'r-',lw=5)
                # fig.show()
                plt.pause(0.1)
                # lines.pop(0)
                ax.lines.remove(lines[0])


if __name__ == '__main__':
    print('start')
    # test1()
    # test_variable()
    test_nn1()
    # ImprovedMethod_Improve(10)
    print('end')
