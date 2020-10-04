import collections
import math
import os
import random
import zipfile

import numpy as np
import urllib.request as request
import tensorflow as tf

dic = {"a": 1, "b": 3, "c": 3}
zip_obj = zip(dic.values(), dic.keys())
dic_obj = dict(zip_obj)

print(zip_obj)
print(dic_obj)

# batch_size = 6
# batch = np.ndarray(shape=(batch_size), dtype=np.int32)
# labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
#
# # print(batch)
# # print(labels)
#
# num_skips = 3
# skip_window = 2
# span = 2 * skip_window + 1
# buffer = collections.deque(maxlen=span)
# data = [5244,
#         3084,
#         12,
#         6,
#         195,
#         2,
#         3134,
#         46,
#         59,
#         156,
#         128,
#         742,
#         477,
#         10634,
#         134,
#         1,
#         27689,
#         2]
# data_index = 0
# for _ in range(span):
#     # 只会保留最后插入的span个变量
#     buffer.append(data[data_index])
#     # 取余数
#     data_index = (data_index + 1) % len(data)
# for i in range(batch_size // num_skips):  # 多少个目标单词
#     # 目标单词
#     target = skip_window
#     targets_to_avoid = [skip_window]
#     # 除了目标之外的词
#     for j in range(num_skips):
#         while target in targets_to_avoid:
#             target = random.randint(0, span - 1)
#         targets_to_avoid.append(target)
#         # 其中一条样本的feature是target
#         batch[i * num_skips + j] = buffer[skip_window]
#         # 标签是语境word,当前的target和buffer中skip_window取到的不同
#         labels[i * num_skips + j, 0] = buffer[target]
#     buffer.append(data[data_index])
#     data_index = (data_index + 1) % len(data)
#
#
# print(batch)
# print('--------------------------')
# print(labels)
# print('--------------------------')
# print(targets_to_avoid)
#
# data=tf.compat.as_str('a big go').split(sep=' ')
# print(data)

a=-1.72133058e-01
print(a)