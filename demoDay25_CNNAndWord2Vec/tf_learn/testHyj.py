# -*- coding: UTF-8 -*-
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

a = -1.72133058e-01
print(a)
doc_word_count_dict = {'1': {'1': 1, '2': 1}, '2': {'3': 1}}
for doc_name, doc_wc in doc_word_count_dict.items():
    doc_wc2 = {}
    doc_len = len(doc_wc)
    for w, c in doc_wc.items():
        doc_wc2[w] = c / doc_len
    doc_word_count_dict[doc_name] = doc_wc2

print(doc_word_count_dict)

arr = [1.1, 2.1, 3.1]
print(1 * arr)


import numpy as np
v1=np.array([1e-1,2.1,2.1])
v2=np.array([3,0,4])
print(0.5*v1)
v3=(v1+v2)
print(v3)
v4=np.array([0]*3)
print(type(v1))
#<class 'numpy.ndarray'>
# print(np.linalg.norm(v1))
# print(np.linalg.norm(v2))
import jieba
from common import file_util
# pre='G:\\bigdata\\badou\\00-data\\data\\'
# out_dir=pre+'out\\'
# std_dir='std\\'
# docs=[]
# docs2=[]
# # 文本索引 k=文本名 v=index
# doc_id_dict={}
# for std_f in file_util.list_file(pre+std_dir):
#     doc_id_dict[std_f]=len(doc_id_dict)
#     with open(pre+std_dir+std_f,mode='r',encoding='utf-8') as f:
#         # docs.append('\n'.join(f.readlines()))
#         docs2.append(f.read())
#     break

# print(docs2[0].split())
# print('------------------------------')
# print(docs[0].split())
# ['银监会', '银行', '可', '减', '持', '或', '全部', '退出', '村镇', '银行', '搜狐', '财经昨日', '中国', '人民', '银行', '中国', '银监会', '中国', '证监会', '中国', '保监会', '上海市', '人民', '政府', '在', '上海', '联合', '召开', '陆家嘴', '论坛', '新闻', '发布会', '来自', '一', '行', '三', '会', '和', '上海市', '政府', '相关', '部门', '的', '负责人', '针对', '目前', '金融', '领域', '的', '热点', '问题', '回答', '了', '记者', '的', '提问', '鼓励', '民营', '资本', '进入', '银行业', '银监会', '办公厅', '副', '主任', '杨少俊', '在', '发布会', '上', '透露', '从', '政策', '上', '而', '言', '村镇', '银行', '在', '经营', '若干年', '之后', '银行业', '的', '持股', '和', '其他', '资本', '持股', '的', '相对', '比例', '是', '可以', '调整', '的', '根据', '运营', '状况', '管理', '状况', '的', '需求', '银行', '可以', '减', '持', '甚至', '全部', '退出', '村镇', '银行', '为了', '落实', '国务院', '新', '条', '我们', '也', '出台', '了', '实施', '细则', '杨少俊', '表示', '在', '这', '一', '细则', '中', '银监会', '将', '村镇', '银行', '主', '发起', '行', '的', '最低', '持股', '比例', '由', '前期', '的', '降低', '到', '并', '鼓励', '民间', '资本', '参与', '农村', '信用社', '的', '改制', '改革', '参与', '中', '小', '金融', '机构', '的', '发起', '设立', '杨少俊', '指出', '除了', '这些', '看', '得到', '的', '支持', '之外', '如果', '仔细', '研读', '政策', '就', '会', '发现', '有', '很', '多', '看', '不', '到', '的', '扶持', '政策', '实际上', '意义', '更为', '重大', '他', '透露', '在', '农村', '的', '一些', '金融', '机构', '改制', '特别', '是', '高', '风险', '机构', '改革', '中', '现在', '就', '允许', '民间', '资本', '控股', '但是', '有', '阶段性', '要求', '从', '实际', '情况', '来', '看', '民间', '资本', '进入', '银行业', '的', '渠道', '还是', '比较', '顺畅', '的', '杨少俊', '表示', '法律', '法规', '上', '从来', '就', '没有', '对', '民间', '资本', '进入', '银行业', '有', '任何', '障碍', '从', '个案', '上来', '讲', '股份制', '银行', '中', '民生银行', '是', '由', '民间', '资本', '主导', '的', '银行', '而', '浙江', '泰隆', '稠', '州', '台州', '等', '不少', '城商行', '民资', '占', '股', '甚至', '高达', '杨少俊', '表示', '民间', '资本', '虽然', '对', '参与', '国有', '商业', '银行', '有', '积极性', '但', '并非', '其', '关注', '重点', '民资', '关注', '更', '多', '的', '是', '中', '小', '商业', '银行', '对于', '股份制', '商业', '银行', '截至', '去年底', '民间', '资本', '占', '比', '达到', '城商行', '方面', '全国', '多', '家中', '民资', '也', '颇', '有', '兴趣', '因为', '可以', '有', '较', '大', '的', '发言权', '甚至', '可以', '控股', '截至', '去年底', '在', '城商行', '总', '股本', '中', '民间', '资本', '占', '比', '达到', '而', '农村', '的', '中', '小', '金融', '机构', '比如', '农', '商行', '农', '合', '社', '等', '民资', '占', '比', '则', '更', '高', '达到', '对', '银行', '存贷', '比', '无', '时点', '要求', '目前', '有', '观点', '指出', '银监会', '对', '银行', '严厉', '的', '存贷', '比', '考核', '要求', '使得', '银行', '的', '放贷', '能力', '受', '到', '制约', '并', '导致', '银行', '对', '存款', '资源', '的', '争夺', '对此', '杨少俊', '表示', '银监会', '对', '存贷', '比', '没有', '考核', '存贷', '比', '实际上', '是', '银行业', '法人', '机构', '内部', '对', '分支', '机构', '做', '的', '一些', '考核', '同时', '也', '有', '一些', '外部', '部门', '有', '一些', '时点', '考核', '的', '问题', '从', '银监会', '来', '说', '从来', '没有', '对', '存贷', '比', '的', '时点', '有', '要求', '我们', '要求', '按照', '商业', '银行法', '对', '存贷', '比', '的', '限制', '是', '这', '是', '一个', '日常', '的', '监管', '考核', '是', '一个', '日常', '的', '监测', '杨少俊', '表示', '他', '指出', '现在', '银行', '冲', '时点', '更', '多', '的', '是', '银行', '自身', '的', '行为', '银监会', '也', '注意', '到', '这', '实际上', '是', '不', '完全', '科学', '的', '所以', '一直', '在', '注重', '引导', '商业', '银行', '优化', '激励', '考核', '机制', '以', '正确', '的', '导向', '去', '对', '分支', '机构', '的', '发展', '做出', '考核', '和', '激励', '最近', '对', '冲', '时点', '做', '了', '一些', '督促', '有', '所', '缓解', '我们', '一直', '希望', '商业', '银行', '按照', '可', '持续', '发展', '的', '要求', '稳健', '地', '发展', '在', '风险', '防范', '上', '认真', '地', '按照', '监管', '部门', '的', '要求', '做好', '风险', '防范', '杨少俊', '表示', '存款', '保险', '制度', '各', '方面', '条件', '具备', '对于', '目前', '业内', '备受', '关注', '的', '存款', '保险', '制度', '央行', '上海', '总部', '副', '主任', '凌涛', '在', '发布会', '上', '表示', '包括', '央行', '在内', '的', '各', '有关', '部门', '对', '这', '一', '问题', '已经', '调研', '了', '很', '长', '时间', '尤其', '是', '这次', '金融', '危机', '以后', '我们', '也', '进一步', '感到', '了', '存款', '保险', '制度', '在', '我们', '国家', '有', '必要', '建立', '和', '推出', '凌涛', '表示', '他', '指出', '从', '前期', '调研', '的', '角度', '看', '存款', '保险', '制度', '建立', '的', '各', '方面', '条件', '已经', '基本', '具备', '这', '也', '是', '参与', '这', '项', '工作', '的', '各', '有关', '部门', '都', '基本', '达成', '的', '共识', '但', '至于', '具体', '什么', '时候', '推出', '还要', '再', '研究', '具体', '的', '条件', '和', '形式', '目前', '有关', '房地产', '信贷', '放松', '的', '预期', '渐', '起', '不少', '地方', '都', '传出', '开发', '贷', '放松', '和', '房贷', '折扣', '持续', '降低', '的', '消息', '对于', '房地产', '信贷', '的', '相关', '政策', '央行', '和', '银监会', '相关', '人士', '在', '发布会', '上', '均', '表示', '目前', '政策', '基调', '与', '前期', '相比', '并', '没有', '任何', '改变', '作者', '唐真龙', '来源', '上海', '证券报分享']
# ['银监会', '银行', '可', '减', '持', '或', '全部', '退出', '村镇', '银行', '搜狐', '财经昨日', '中国', '人民', '银行', '中国', '银监会', '中国', '证监会', '中国', '保监会', '上海市', '人民', '政府', '在', '上海', '联合', '召开', '陆家嘴', '论坛', '新闻', '发布会', '来自', '一', '行', '三', '会', '和', '上海市', '政府', '相关', '部门', '的', '负责人', '针对', '目前', '金融', '领域', '的', '热点', '问题', '回答', '了', '记者', '的', '提问', '鼓励', '民营', '资本', '进入', '银行业', '银监会', '办公厅', '副', '主任', '杨少俊', '在', '发布会', '上', '透露', '从', '政策', '上', '而', '言', '村镇', '银行', '在', '经营', '若干年', '之后', '银行业', '的', '持股', '和', '其他', '资本', '持股', '的', '相对', '比例', '是', '可以', '调整', '的', '根据', '运营', '状况', '管理', '状况', '的', '需求', '银行', '可以', '减', '持', '甚至', '全部', '退出', '村镇', '银行', '为了', '落实', '国务院', '新', '条', '我们', '也', '出台', '了', '实施', '细则', '杨少俊', '表示', '在', '这', '一', '细则', '中', '银监会', '将', '村镇', '银行', '主', '发起', '行', '的', '最低', '持股', '比例', '由', '前期', '的', '降低', '到', '并', '鼓励', '民间', '资本', '参与', '农村', '信用社', '的', '改制', '改革', '参与', '中', '小', '金融', '机构', '的', '发起', '设立', '杨少俊', '指出', '除了', '这些', '看', '得到', '的', '支持', '之外', '如果', '仔细', '研读', '政策', '就', '会', '发现', '有', '很', '多', '看', '不', '到', '的', '扶持', '政策', '实际上', '意义', '更为', '重大', '他', '透露', '在', '农村', '的', '一些', '金融', '机构', '改制', '特别', '是', '高', '风险', '机构', '改革', '中', '现在', '就', '允许', '民间', '资本', '控股', '但是', '有', '阶段性', '要求', '从', '实际', '情况', '来', '看', '民间', '资本', '进入', '银行业', '的', '渠道', '还是', '比较', '顺畅', '的', '杨少俊', '表示', '法律', '法规', '上', '从来', '就', '没有', '对', '民间', '资本', '进入', '银行业', '有', '任何', '障碍', '从', '个案', '上来', '讲', '股份制', '银行', '中', '民生银行', '是', '由', '民间', '资本', '主导', '的', '银行', '而', '浙江', '泰隆', '稠', '州', '台州', '等', '不少', '城商行', '民资', '占', '股', '甚至', '高达', '杨少俊', '表示', '民间', '资本', '虽然', '对', '参与', '国有', '商业', '银行', '有', '积极性', '但', '并非', '其', '关注', '重点', '民资', '关注', '更', '多', '的', '是', '中', '小', '商业', '银行', '对于', '股份制', '商业', '银行', '截至', '去年底', '民间', '资本', '占', '比', '达到', '城商行', '方面', '全国', '多', '家中', '民资', '也', '颇', '有', '兴趣', '因为', '可以', '有', '较', '大', '的', '发言权', '甚至', '可以', '控股', '截至', '去年底', '在', '城商行', '总', '股本', '中', '民间', '资本', '占', '比', '达到', '而', '农村', '的', '中', '小', '金融', '机构', '比如', '农', '商行', '农', '合', '社', '等', '民资', '占', '比', '则', '更', '高', '达到', '对', '银行', '存贷', '比', '无', '时点', '要求', '目前', '有', '观点', '指出', '银监会', '对', '银行', '严厉', '的', '存贷', '比', '考核', '要求', '使得', '银行', '的', '放贷', '能力', '受', '到', '制约', '并', '导致', '银行', '对', '存款', '资源', '的', '争夺', '对此', '杨少俊', '表示', '银监会', '对', '存贷', '比', '没有', '考核', '存贷', '比', '实际上', '是', '银行业', '法人', '机构', '内部', '对', '分支', '机构', '做', '的', '一些', '考核', '同时', '也', '有', '一些', '外部', '部门', '有', '一些', '时点', '考核', '的', '问题', '从', '银监会', '来', '说', '从来', '没有', '对', '存贷', '比', '的', '时点', '有', '要求', '我们', '要求', '按照', '商业', '银行法', '对', '存贷', '比', '的', '限制', '是', '这', '是', '一个', '日常', '的', '监管', '考核', '是', '一个', '日常', '的', '监测', '杨少俊', '表示', '他', '指出', '现在', '银行', '冲', '时点', '更', '多', '的', '是', '银行', '自身', '的', '行为', '银监会', '也', '注意', '到', '这', '实际上', '是', '不', '完全', '科学', '的', '所以', '一直', '在', '注重', '引导', '商业', '银行', '优化', '激励', '考核', '机制', '以', '正确', '的', '导向', '去', '对', '分支', '机构', '的', '发展', '做出', '考核', '和', '激励', '最近', '对', '冲', '时点', '做', '了', '一些', '督促', '有', '所', '缓解', '我们', '一直', '希望', '商业', '银行', '按照', '可', '持续', '发展', '的', '要求', '稳健', '地', '发展', '在', '风险', '防范', '上', '认真', '地', '按照', '监管', '部门', '的', '要求', '做好', '风险', '防范', '杨少俊', '表示', '存款', '保险', '制度', '各', '方面', '条件', '具备', '对于', '目前', '业内', '备受', '关注', '的', '存款', '保险', '制度', '央行', '上海', '总部', '副', '主任', '凌涛', '在', '发布会', '上', '表示', '包括', '央行', '在内', '的', '各', '有关', '部门', '对', '这', '一', '问题', '已经', '调研', '了', '很', '长', '时间', '尤其', '是', '这次', '金融', '危机', '以后', '我们', '也', '进一步', '感到', '了', '存款', '保险', '制度', '在', '我们', '国家', '有', '必要', '建立', '和', '推出', '凌涛', '表示', '他', '指出', '从', '前期', '调研', '的', '角度', '看', '存款', '保险', '制度', '建立', '的', '各', '方面', '条件', '已经', '基本', '具备', '这', '也', '是', '参与', '这', '项', '工作', '的', '各', '有关', '部门', '都', '基本', '达成', '的', '共识', '但', '至于', '具体', '什么', '时候', '推出', '还要', '再', '研究', '具体', '的', '条件', '和', '形式', '目前', '有关', '房地产', '信贷', '放松', '的', '预期', '渐', '起', '不少', '地方', '都', '传出', '开发', '贷', '放松', '和', '房贷', '折扣', '持续', '降低', '的', '消息', '对于', '房地产', '信贷', '的', '相关', '政策', '央行', '和', '银监会', '相关', '人士', '在', '发布会', '上', '均', '表示', '目前', '政策', '基调', '与', '前期', '相比', '并', '没有', '任何', '改变', '作者', '唐真龙', '来源', '上海', '证券报分享']

# doc_word_count_dict=file_util.read_obj(out_dir+'doc_word_count_dict_std.txt')
# for doc_name,doc_wc in doc_word_count_dict.items():
#     if '八第' in doc_wc or '程序化' in doc_wc:
#         print(doc_name)

s='民航部门又出排堵保畅新招月日起京沪穗三地四大机场航班统一放行将正式运行意味着在这几条航线上延误未定的情况将有所减少民航部门可提供一个相对合理的预计起飞时刻'

jieba.add_word('现用')
cut_s=[i for i in jieba.cut(s,HMM=True)]
print(cut_s)
print('-----------------')
# import thulac
# th=thulac.thulac(seg_only=True)
# th_res=th.cut(s,text=True)
# print(th_res)
# from snownlp import SnowNLP
#
# s1 = SnowNLP(s)
# print(s1.words)

# tokenizer = hanlp.load('LARGE_ALBERT_BASE')
from common import common_util
# pre='G:\\bigdata\\badou\\00-data\\word2vec\\test\\1business.seg.cln_out2.txt'
# lines=file_util.read_file(pre)
# for line in lines:
#     line1=line.replace(' ','')
#     print(line1)
#     cut_line=jieba.cut(line1)
#     print([i for i in cut_line])
import pandas as pd
pre='F:\\八斗学院\\视频\\14期正式课\\00-data\\nn\\'
df_file=pre+'music_data.csv'
print('df to csv done')
# chunksize 参数，用以指定一个块大小(每次读取多少行)
chunks = pd.read_csv(df_file,iterator = True,chunksize=1000)
for chunk in chunks:
    print(chunk.shape)

chunk = chunks.get_chunk(5)
print(chunk)
print(np.array(chunk))


