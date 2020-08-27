from __future__ import print_function
# Ignore all GPUs, tf random forest does not benefit from it.
import os
import tensorflow as tf
from tensorflow.contrib.tensor_forest.python import tensor_forest
from tensorflow.python.ops import resources

os.environ["CUDA_VISIBLE_DEVICES"] = ""
# 导入 MNIST 数据
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("./tmp/data/", one_hot=False)
# 参数
num_steps = 500  # Total steps to train
batch_size = 1024  # 每批处理样本数
num_classes = 10  # 10个数字=>10个分类
num_features = 784  # 每张图片 28x28 像素 => 784特征
num_trees = 10
max_nodes = 1000
# 输入数据
X = tf.placeholder(tf.float32, shape=[None, num_features])
# 用数字表示随机森林中的标签(类id)
Y = tf.placeholder(tf.int32, shape=[None])
# 随机森林参数
hparams = tensor_forest.ForestHParams(num_classes=num_classes,
                                      num_features=num_features,
                                      num_trees=num_trees,
                                      max_nodes=max_nodes).fill()
# 建立随机森林
forest_graph = tensor_forest.RandomForestGraphs(hparams)
# 获取训练图，计算损失率
train_op = forest_graph.training_graph(X, Y)
loss_op = forest_graph.training_loss(X, Y)
# 计算准确率
infer_op, _, _ = forest_graph.inference_graph(X)
correct_prediction = tf.equal(tf.argmax(infer_op, 1), tf.cast(Y, tf.int64))
accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# 初始化变量和森林资源
init_vars = tf.group(tf.global_variables_initializer(),
                     resources.initialize_resources(resources.shared_resources()))
# 启动TensorFlow会话
sess = tf.Session()
# 初始化
sess.run(init_vars)
# 训练
for i in range(1, num_steps + 1):
    # 获取一批图片数据
    batch_x, batch_y = mnist.train.next_batch(batch_size)
    _, l = sess.run([train_op, loss_op], feed_dict={X: batch_x, Y: batch_y})
    if i % 50 == 0 or i == 1:
        acc = sess.run(accuracy_op, feed_dict={X: batch_x, Y: batch_y})
        print('Step %i, Loss: %f, Acc: %f' % (i, l, acc))
# 测试模型
test_x, test_y = mnist.test.images, mnist.test.labels
print("Test Accuracy:", sess.run(accuracy_op, feed_dict={X: test_x, Y: test_y}))
