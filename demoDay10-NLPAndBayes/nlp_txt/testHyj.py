import re
from string import punctuation
from common import common_util
"""
去除中英文标点符号
"""
text = "he is good, Hello, world! 这里，苹果：我;第!一个程序\?()（）<>《》【】 "+"\sA-Za-z～()（）【】%*#+-\.\\\/:=：__,，。、;；“”""''’‘？?！!<《》>^&{}|=……"
punc2=r"""~`!#$%^&*()_+-=|\'\\;":/.,?><~·！@#￥%……&*（）——+-=“：’；、。，？》《{}!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~“”？，！【】（）、。：；’‘……￥·"""
text=text+punc2
print(common_util.trim_with_space_flag(text,True))
# punctuation =punctuation+ r"""~`!#$%^&*()_+-=|\'\\;":/.,?><~·！@#￥%……&*（）——+-=“：’；、。，？》《{}!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~“”？，！【】（）、。：；’‘……￥·"""
# s =text
# punc = punctuation + r'··.,;\\《》？！“”‘’@#￥%…&×（）——+【】{};；●，。&～、|\s:：'
# dicts={i:'' for i in punc}
# punc_table=str.maketrans(dicts)
# new_s=s.translate(punc_table)
print(common_util.trim_with_str_translate(text))

# pattern=re.compile(common_util.punc2)
res=re.sub(r"[{}]+".format(common_util.punc2), '', text)
print(res)

# text= re.sub(r"[{}]+".format(punc)," ",text)
# print(text)

# file_pre='F:\\00-data\\data\\'
# file1=file_pre+'1business.seg.cln.txt'
# with open(file1,mode='r',encoding='utf-8') as f:
#     content=f.read()
# # 去除标点
# content=common_util.trim_with_str_translate(content)
# from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer,TfidfTransformer
# corpus = ["我 来到 北京  清华大学",
#           "他 来到 了 网易 杭研 大厦",
#           "小明 硕士 毕业 与 中国 科学院",
#           "我 爱 北京 天安门"]
#
# #将文本中的词语转换为词频矩阵
# vectorizer = CountVectorizer(token_pattern='\\b\\w+\\b')
# #计算个词语出现的次数
# X = vectorizer.fit_transform(corpus)
# word = vectorizer.get_feature_names()
# print(word)
# #查看词频结果
# print('X.toarray():',X.toarray())
#
#
# # norm=None对词频结果不归一化
# # use_idf=False, 因为使用的是计算tfidf的函数, 所以要忽略idf的计算
# '''
# norm='l1'表示 x/(sum(abs(x)) 即x/各特征的绝对值之和
# norm='l2'表示 x/sqrt(sum(x^2)) 即x/sqrt(各特征的平方和)
# '''
# transformer = TfidfTransformer( use_idf=True,norm=None)
# tf = transformer.fit_transform(X)
# word = vectorizer.get_feature_names()
# weight = tf.toarray()
#
# print(weight)
# print('----------------------------------------------------')

# tfidfvec=TfidfVectorizer(token_pattern='\\b\\w+\\b',norm='l2')
# res=tfidfvec.fit_transform(corpus)
# print(res.toarray())