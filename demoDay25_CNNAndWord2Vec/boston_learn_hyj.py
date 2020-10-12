import tensorflow as tf
from sklearn.datasets import load_boston
batch_size=10
number_feats=14
boston=load_boston()

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