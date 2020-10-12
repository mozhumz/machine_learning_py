'''
基于文本相似度进行推荐
1 计算每篇文章各词的wordvec=tfidf*word2vec
语料库为2454篇新闻文章
tf=词频/文章的总词数
idf=文章总数/（出现该词的文章数+1）
2 计算文章的向量docvec=sum(wordvec)/len(set(word))
3 计算文本相似度 取top-N20
'''
from common import file_util,common_util
import math
from demoDay25_CNNAndWord2Vec import word2vec_hyj
import numpy as np

pre='G:\\bigdata\\badou\\00-data\\data\\'
out_dir=pre+'out\\'
std_dir='std\\'
common_util.mkdirs(out_dir,False)
'''1 计算wordcount'''
# 遍历所有文本，对每个文本处理成标准的分词文本（不含标点符号）
# if len(file_util.list_file(pre+std_dir))==0:
for f_name in file_util.list_file(pre,True):
    print('deal f_name:',f_name)
    lines=file_util.get_format_lines(f_name,pre)
    if lines:
        f_dir=pre+std_dir
        common_util.mkdirs(f_dir,False)
        with open(f_dir+f_name,mode='w',encoding='utf-8') as f:
            # 存储标准分词文本
            f.writelines([line+'\n' for line in lines])

# 读取文本
docs={}
# 文本索引 k=文本名 v=index
doc_id_dict={}
for std_f in file_util.list_file(pre+std_dir):
    doc_id_dict[std_f]=len(doc_id_dict)
    with open(pre+std_dir+std_f,mode='r',encoding='utf-8') as f:
        docs[std_f]='\n'.join(f.readlines())
        # docs.append('\n'.join(f.readlines()))
#
file_util.save_obj(doc_id_dict,out_dir+'doc_id_dict.txt')
# # 词表
word_id_dict={}
# k=doc_name v={k=word,v=count}
doc_word_count_dict={}
for document_name,document in docs.items():
    doc_word_count_dict[document_name]={}
    for word in document.split():
        # 每个文本的wordcount
        if word not in doc_word_count_dict[document_name]:
            doc_word_count_dict[document_name][word]=0
        doc_word_count_dict[document_name][word]+=1
        # 构建词表
        if word not in word_id_dict:
            word_id_dict[word]=len(word_id_dict)

# 存储wordcount文本
file_util.save_obj(doc_word_count_dict,out_dir+'doc_word_count_dict.txt')
file_util.save_obj(word_id_dict,out_dir+'word_id_dict.txt')

# doc_word_count_dict=file_util.read_obj(out_dir+'doc_word_count_dict.txt')
# 词频标准化 词频/文本总词数
# 统计出现某单词的文档数
doc_freq={}
for doc_name,doc_wc in doc_word_count_dict.items():
    doc_len=len(doc_wc)
    for w,c in doc_wc.items():
        doc_wc[w]=c/doc_len
        if w not in doc_freq:
            doc_freq[w]=0
        doc_freq[w]+=1

# 存储wordcount-标准化词频后的文本
file_util.save_obj(doc_word_count_dict,out_dir+'doc_word_count_dict_std.txt')
file_util.save_obj(doc_freq,out_dir+'doc_freq.txt')
'''2 tf*idf*word2vec'''
model_pre='G:\\bigdata\\badou\\00-data\\word2vec\\'
model_file_path = model_pre+'out\\cbow\\wiki_cbow_10.model'
model_file_path2 = model_pre+'out\\cbow\\fast_cbow_10.model'
model=word2vec_hyj.load_model(model_file_path2)
# 文档总数
doc_id_dict=file_util.read_obj(out_dir+'doc_id_dict.txt')
doc_total=len(doc_id_dict)
doc_freq=file_util.read_obj(out_dir+'doc_freq.txt')
doc_vec={}
doc_word_count_dict=file_util.read_obj(out_dir+'doc_word_count_dict_std.txt')
vec_size=model.wv.vector_size
for doc_name,doc_wc in doc_word_count_dict.items():
    v_sum=np.array([0.0]*vec_size)
    for w,c in doc_wc.items():
        # tf*idf*word2vec
        v_sum+=c*math.log10(doc_total/(1+doc_freq[w]))*(model[w])
    # 计算文章的向量docvec=sum(wordvec)/len(set(word))
    doc_vec[doc_name]=v_sum/len(doc_wc)

file_util.save_obj(doc_vec,out_dir+'doc_vec.txt')
'''5 用l1或l2标准化文本向量docvec'''

'''6 计算文本相似度 取top-N20'''
v_d1=doc_vec['1business.seg.cln.txt']
v_d2=doc_vec['1yule.seg.cln.txt']
sim=np.dot(v_d1,v_d2)/(np.linalg.norm(v_d1)*np.linalg.norm(v_d2))
print(v_d1)
print(v_d2)
print(sim)