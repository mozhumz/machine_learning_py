import tensorflow as tf
x = tf.constant([[1., 1.], [2., 2.]])
ss=tf.reduce_mean(x)
with tf.Session() as sess:

    print(sess.run(ss))
