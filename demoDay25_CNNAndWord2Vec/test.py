# -*- coding: UTF-8 -*-
import tensorflow as tf
# import tensorflow.contrib.eager as tfe
# tfe.enable_eager_execution()

# a = tf.constant([1,2,1,2,3,0])
# a = tf.reshape(a,[2,3])
# b = tf.constant([2])
# c = tf.reduce_max(a,axis=0,keep_dims=True)

# init = tf.global_variables_initializer()
#
# with tf.Session() as sess:
#     sess.run(init)
#     print(a.eval())
#     print(sess.run(c))

# from tensorflow.examples.tutorials.mnist import input_data
# import tensorflow as tf
import numpy as np
#
# mnist = input_data.read_data_sets('MNIST/', one_hot=True)
# a = mnist.train.images
# sample = np.array(a[0])
# print(sample.reshape([28,28]))
#
# from pprint import pprint
# pprint(a[0])



# # data，以后做好数据处理的数据，从文件中读入数据
# x_data = np.linspace(-1,1,200).reshape([200,1])
# y_data = x_data*0.5+0.3
#
# # 在session中占位，位置站住，为的是在训练的时候可以把数据传入
# x = tf.placeholder(tf.float32,[None,1])
# y_true = tf.placeholder(tf.float32,[None,1])
#
# # 定义模型参数变量
# W = tf.Variable(tf.zeros([1,1]))
# b = tf.Variable(tf.zeros([1]))
#
# # 表示输出的计算方式
# y_pred = tf.matmul(x,W)+b
#
# # 目标函数：cost，loss，我们在寻找是的这个loss最小的w，b
# loss = tf.reduce_mean(tf.square(y_pred-y_true))
# # 通过梯度方式寻找是的loss最小的w，b
# optimizer = tf.train.GradientDescentOptimizer(0.2).minimize(loss)
#
# init = tf.global_variables_initializer()
#
# with tf.Session() as sess:
#     sess.run(init)
#     for step in range(200):
#         sess.run(optimizer, feed_dict={x: x_data, y_true: y_data})
#         if step%20 == 0:
#             print(sess.run([W, b], feed_dict={x: x_data, y_true: y_data}))

# [1 0 1]
a = [
    [1, 2],  # 1
    [2, 1],  # 0
    [1, 3]]  # 1


# with tf.Session() as sess:
#     print(sess.run(tf.argmax(a, axis=0)))

# b=(0,)
# print(b)
# print(a[-1])

arr=np.array([[1,2,3,4,5],[10,21,31,41,15],[101,211,311,141,115]])
print(arr[:,3])

# import matplotlib.pyplot as plt
# plt.plot([1,2,3,4])
# plt.ylabel('some numbers')
# plt.show()

t1=(1,2,3)
print(t1[1])
print('------------------')
x = np.arange(9).reshape(3, 3)
y = np.ones([3, 3])
print(x)
print(y)
print('------------------')
c = np.r_[x, y]

d = np.c_[x, y]

# print(d)
# print(c.round(1))
a1=np.array([[1.51,2.15],[3.25,4.67]])
# print(a1[1,:])
# print(a1.round(1))
# a2=np.reshape([1,2,3],[1,3])
# a3=np.reshape([1,2,3],[3,1])
# print(np.matmul(a2,a3)[0][0])
a_list=[1,2,3]
# a_list+=1
# print(a_list)
# a_list+=[2]
# print(a_list)
x = tf.constant([[1, 2, 3], [1, 1, 1]])
y=np.array([[0,0,1],[0,0,1]])
y_hat=np.array([[0.1,0.1,0.8],[0.02,0.08,0.9]])
# loss=tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=y_hat)
# cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(y_hat),reduction_indices=1))
# with tf.Session() as sess:
#     print(sess.run([loss,cross_entropy]))

a = [[1,2,3],[4,5,6]]
b = [[1,0,3],[1,5,1]]
correct_pred=tf.equal(tf.argmax(a,axis=1),tf.argmax(b,axis=1))

# with tf.Session() as sess:
#     print(sess.run(tf.argmax(a,axis=1)))
#     out=sess.run(correct_pred)
#     pred_cast=tf.cast(out, tf.float32)
#     print(out,sess.run(pred_cast))
# eta=tf.constant(0.1)
# delta_w_2=tf.constant(0.2,shape=[30,10])
def softmax(x):
    sum_raw = np.sum(np.exp(x), axis=-1)
    x1 = np.ones(np.shape(x))
    for i in range(np.shape(x)[0]):
        x1[i] = np.exp(x[i]) / sum_raw[i]
    return x1
loss=tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=y_hat))
loss2 = (-tf.reduce_sum(y*tf.log(tf.clip_by_value(softmax(y_hat),1e-10,1.0))))
loss3=tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y,logits=y_hat))
# with tf.Session() as sess:
#     # print(sess.run(tf.multiply(eta,delta_w_2)))
#     print(sess.run(loss))
#     print(sess.run(loss3))
    # print(sess.run(tf.nn.relu(y_hat)))

arr=np.array([[1,2,3,4],[11,22,33,44],[22,33,55,66]])
print(arr[:1])

# print(arr.argmax(axis=1))
# print(arr.argmax(axis=0))

arr2=np.array([
    [[0,1,2],
     [2,3,4],
     ],
    [[10,11,12],
     [12,13,14],
     ],
    [[20,21,22],
     [22,23,24],
     ]
])

print(arr2.argmax())
print(arr2.argmax(axis=0))
print(arr2.argmax(axis=1))