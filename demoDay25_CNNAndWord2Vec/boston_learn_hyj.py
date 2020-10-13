import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def normalize(X):
    mean=np.mean(X)
    std=np.std(X)
    X=(X-mean)/std
    return X

(x_train, y_train), (x_test, y_test)=tf.keras.datasets.boston_housing.load_data()
x_train=x_train[:,5]
x_test=x_test[:,5]
print('x_train:',x_train[:5])
print('y_train:',y_train[:5])
n_samples=len(x_train)
print('n_samples:',n_samples)

X=tf.placeholder(tf.float32,name='X')
Y=tf.placeholder(tf.float32,name='Y')

# 声明 w b
b=tf.Variable(0.0)
w=tf.Variable(0.0)
# 模型
Y_hat=X*w+b
loss=tf.square(Y-Y_hat,name='loss')
optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)
init_op=tf.global_variables_initializer()
total=[]
with tf.Session() as sess:
    sess.run(init_op)
    writer=tf.summary.FileWriter('graphs',sess.graph)
    # 迭代100次
    for i in range(100):
        total_loss=0
        for x,y in zip(x_train,y_train):
            _,l=sess.run([optimizer,loss],feed_dict={X:x,Y:y})
            print('optimizer,loss:',_,l)
            total_loss+=l
        total.append(total_loss/n_samples)
        print('epoch {0}: Loss {1}'.format(i,total_loss/n_samples))
    writer.close()
    b_val,w_val=sess.run([b,w])

y_pred=x_train*w_val+b_val
print('done')
plt.plot(x_train,y_train,'bo',label='real_y')
plt.plot(x_train,y_pred,'r',label='pred_y')
plt.legend()
plt.show()
plt.plot(total)
plt.show()


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