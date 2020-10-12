# tensorflow 基础学习
import tensorflow as tf
import numpy as np

# a=tf.constant([2,3])
# b=tf.constant([4,5])
# c=a*b
#
# print(c)
#
# sess=tf.Session()
# print(sess.run(a))
# print(sess.run([a,b]))
# print(sess.run(c))
# sess.close()
# w = tf.Variable(tf.random_normal([1,10]))

# 要初始化后才能计算
# input_data = tf.Variable(1,dtype=np.float32)
# input_data2 = tf.Variable(1,dtype=np.float32)
# result = tf.add(input_data,input_data2)
# init = tf.global_variables_initializer()
# sess = tf.Session()
# print(sess.run(init))
# print(sess.run(result))
#2.0

# input1 = tf.placeholder(tf.float32)
# input2 = tf.placeholder(tf.float32)
# output = tf.multiply(input1,input2)
# sess = tf.Session()
# print(sess.run([output],feed_dict={input1:[5.,2],input2:[8,3]}))
# print(output)
def test3():
    x=tf.placeholder(np.float32)
    y=2*x
    data=tf.random_uniform([4,5],10)
    with tf.Session() as sess:
        xdata=sess.run(data)
        print(xdata)
        print('----------------------')
        res=sess.run(y,feed_dict={x:xdata})
        print(res)

def test4():
    '''InteractiveSession 使自己成为默认会话，因此你可以使用 eval() 直接调用运行张量对象而不用显式调用会话'''
    sess=tf.InteractiveSession()
    # 创建单位矩阵 5*5
    i_matrix=tf.eye(5)
    # print(i_matrix.eval())
    x=tf.Variable(tf.eye(10))
    # 初始化变量x
    x.initializer.run()
    # print(x.eval())
    a=tf.Variable(tf.random_uniform([5,10],1,5))
    a.initializer.run()
    # 矩阵相乘a*x
    product=tf.matmul(a,x)
    # print(product.eval())
    b=tf.Variable(tf.random_uniform([5,10],0,2,dtype=tf.int32))
    b.initializer.run()
    # print(b.eval())
    b_new=tf.cast(b,dtype=tf.float32)
    # print(b_new.eval())
    t_sum=tf.add(product,b_new)
    t_sub=product-b_new
    print(t_sum.eval())
    print(t_sub.eval())

def test5():
    a=tf.Variable(tf.random_normal([4,5],stddev=2))
    b=tf.Variable(tf.random_normal([4,5],stddev=2))
    con=tf.constant(2,dtype=tf.float32)
    A=a*b
    B=tf.scalar_mul(2,a)
    C=tf.div(a,b)
    D=tf.mod(a,b)
    # 变量的定义将指定变量如何被初始化，但是必须显式初始化所有的声明变量。在计算图的定义中通过声明初始化操作对象来实现
    init_op=tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_op)
        # writer=tf.summary.FileWriter('graphs',sess.graph)
        # print(sess.run([a,b,A,B,C,D]))
        print(sess.run([a,B]))
    # writer.close()



if __name__ == '__main__':
    test5()