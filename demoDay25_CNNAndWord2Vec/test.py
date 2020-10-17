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
y=[0,0,1]
y_hat=[0.1,0.1,0.8]
loss=tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=y_hat)
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(y_hat),reduction_indices=1))
with tf.Session() as sess:
    print(sess.run([loss,cross_entropy]))