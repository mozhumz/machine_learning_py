# tensorflow 基础学习
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

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





'''
TensorFlow中读取图像数据的三种方式
'''
def read_image(file_name):
    '''
    1 处理单张图片
    :param file_name:
    :return:
    '''
    img = tf.read_file(filename=file_name)   #默认读取格式为uint8
    print("img 的类型是",type(img));
    img = tf.image.decode_jpeg(img,channels=0) # channels 为1得到的是灰度图，为0则按照图片格式来读
    return img

def read_image_test( ):
    with tf.device("/cpu:0"):
        img_path='./1.jpg'
        img=read_image(img_path)
        with tf.Session() as sess:
            image_numpy=sess.run(img)
            print(image_numpy)
            print(image_numpy.dtype)
            print(image_numpy.shape)
            plt.imshow(image_numpy)
            plt.show()

def get_image_batch(data_file,batch_size):
    '''
    2 读取大量图像用于训练
    :param data_file: 文件夹所在的路径 不包括文件名
    :param batch_size:
    :return:
    '''
    # 遍历指定目录下的文件名称，存放到一个list中。当然这个做法有很多种方法，比如glob.glob，或者tf.train.match_filename_once
    data_names=[os.path.join(data_file,k) for k in os.listdir(data_file)]

    #这个num_epochs函数在整个Graph是local Variable，所以在sess.run全局变量的时候也要加上局部变量。
    filenames_queue=tf.train.string_input_producer(data_names,num_epochs=50,shuffle=True,capacity=512)
    reader=tf.WholeFileReader()
    _,img_bytes=reader.read(filenames_queue)
    image=tf.image.decode_png(img_bytes,channels=1)    #读取的是什么格式，就decode什么格式
    #解码成单通道的，并且获得的结果的shape是[?, ?,1]，也就是Graph不知道图像的大小，需要set_shape
    image.set_shape([180,180,1])   #set到原本已知图像的大小。或者直接通过tf.image.resize_images
    image=tf.image.convert_image_dtype(image,tf.float32)
    # image=tf.image.resize_images(image,(180,180))
    #预处理  下面的一句代码可以换成自己想使用的预处理方式
    #image=tf.divide(image,255.0)
    return tf.train.batch([image],batch_size)


def get_image_batch_test( ):
    img_path=r'F:\dataSet\WIDER\WIDER_train\images\6--Funeral'  #本地的一个数据集目录，有足够的图像
    img=get_image_batch(img_path,batch_size=10)
    image=img[0]  #取出每个batch的第一个数据
    print(image)
    init=[tf.global_variables_initializer(),tf.local_variables_initializer()]
    with tf.Session() as sess:
        sess.run(init)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess,coord=coord)
        try:
            while not coord.should_stop():
                # 省略对image的使用，如果仅仅执行下面的代码，image始终是同一个image。我们需要
                # sess.run来实现image的迭代，感谢monk1992的指正
                print(image.shape)
        except tf.errors.OutOfRangeError:
            print('read done')
        finally:
            coord.request_stop()
        coord.join(threads)


'''
3 对TFRecorder解码获得图像数据
'''
import glob
import os
from PIL import Image

def _int64_feature(value):
    if not isinstance(value,list):
        value=[value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))
def _byte_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

#从图片路径读取图片编码成tfrecord
def encode_to_tfrecords(data_path,name,rows=24,cols=16):
    folders=os.listdir(data_path)#这里都和我图片的位置有关
    folders.sort()
    numclass=len(folders)
    i=0
    npic=0
    writer=tf.python_io.TFRecordWriter(name)
    for floder in folders:
        path=data_path+"/"+floder
        img_names=glob.glob(os.path.join(path,"*.bmp"))
        for img_name in img_names:
            img_path=img_name
            img=Image.open(img_path).convert('P')
            img=img.resize((cols,rows))
            img_raw=img.tobytes()
            labels=[0]*34#我用的是softmax，要和预测值的维度一致
            labels[i]=1
            example=tf.train.Example(features=tf.train.Features(feature={#填充example
                'image_raw':_byte_feature(img_raw),
                'label':_int64_feature(labels)}))
            writer.write(example.SerializeToString())#把example加入到writer里，最后写到磁盘。
            npic=npic+1
        i=i+1
    writer.close()

# 解码tfrecord
def decode_from_tfrecord(filequeuelist,rows=24,cols=16):
    reader=tf.TFRecordReader()#文件读取
    _,example=reader.read(filequeuelist)
    features=tf.parse_single_example(example,features={'image_raw':#解码
                                                           tf.FixedLenFeature([],tf.string),
                                                       'label':tf.FixedLenFeature([34,1],tf.int64)})
    image=tf.decode_raw(features['image_raw'],tf.uint8)
    image.set_shape(rows*cols)
    image=tf.cast(image,tf.float32)*(1./255)-0.5
    label=tf.cast(features['label'],tf.int32)
    return image,label


def get_batch(filename_queue,batch_size):
    with tf.name_scope('get_batch'):
        [image,label]=decode_from_tfrecord(filename_queue)
        images,labels=tf.train.shuffle_batch([image,label],batch_size=batch_size,num_threads=2,
                                             capacity=100+3*batch_size,min_after_dequeue=100)
        return images,labels


def generate_filenamequeue(filequeuelist):
    filename_queue=tf.train.string_input_producer(filequeuelist,num_epochs=5)
    return filename_queue

def test_tfrecord(filename,batch_size):
    filename_queue=generate_filenamequeue(filename)
    [images,labels]=get_batch(filename_queue,batch_size)
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())
    sess=tf.InteractiveSession()
    sess.run(init_op)
    coord=tf.train.Coordinator()
    threads=tf.train.start_queue_runners(sess=sess,coord=coord)
    i=0
    try:
        while not coord.should_stop():
            image,label=sess.run([images,labels])
            i=i+1
            if i%1000==0:
                for j in range(batch_size):#之前tfrecord编码的时候，数据范围变成[-0.5,0.5],现在相当于逆操作，把数据变成图片像素值
                    image[j]=(image[j]+0.5)*255
                    ar=np.asarray(image[j],np.uint8)
                    #image[j]=tf.cast(image[j],tf.uint8)
                    img=Image.frombytes("P",(16,24),ar.tostring())#函数参数中宽度高度要注意。构建24×16的图片
                    img.save("/home/wen/MNIST_data/reverse_%d.bmp"%(i+j),"BMP")#保存部分图片查看
            '''if(i>710):
                print("step %d"%(i))
                print image
                print label'''
    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        # When done, ask the threads to stop.
        coord.request_stop()

    # Wait for threads to finish.
    coord.join(threads)
    sess.close()




if __name__ == '__main__':
    test5()