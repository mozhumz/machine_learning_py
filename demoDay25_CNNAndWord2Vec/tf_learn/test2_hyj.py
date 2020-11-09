from tensorflow.python.client import device_lib
# 测试tensorflow安装成功与否
import tensorflow as tf
import numpy as np
import math
print(tf.test.is_gpu_available())

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

print(get_available_gpus())

'''
softmax 交叉熵公式验证 -sum(yi*ln(ai)) yi为样本i的真实标签=1 ai=(softmax(yi_hat)[max(yi)]) 即取yi对应下标的值 
'''
def softmax(x):
    sum_raw = np.sum(np.exp(x), axis=-1)
    x1 = np.ones(np.shape(x))
    for i in range(np.shape(x)[0]):
        x1[i] = np.exp(x[i]) / sum_raw[i]
    return x1

def get_loss(y:np.array([[]]),y_hat:np.array([[]])):
    res=0.
    mat_val=softmax(y_hat)
    print('mat_val:',mat_val)
    # sum所有元素求和
    res=np.sum(y*np.log(mat_val))
    return res

y=np.array([[0,1,0],[0,1,0]])
y_hat=np.array([[0.9,0.1,1],[0.2,0.8,2]])
print(np.argmax(y,axis=1))
print(get_loss(y,y_hat))
loss=tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=y_hat))





with tf.Session() as sess:
    loss_val=sess.run(loss)
    print(loss_val)