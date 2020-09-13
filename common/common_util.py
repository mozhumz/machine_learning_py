import os
import re
from string import punctuation
import math

# 标点符号集合
punc = punctuation + r'··.,;\\《》？！“”‘’@#￥%…&×（）——+【】{};；●，。&～、|\s:：'
dicts = {i: '' for i in punc}


# 根据文件路径（相对或绝对）创建目录
def mkdirs(file):
    dir = os.path.dirname(os.path.abspath(file))
    print(dir)
    if not os.path.exists(dir):
        os.makedirs(dir)


def trim_with_space_flag(text, space_flag=True):
    """
    去除标点，用空格替换
    """
    if text is None:
        return ''
    replace = " " if space_flag == True else ""
    return re.sub(r"[{}]+".format(punc), replace, text)


def trim_with_str_translate(text):
    """
    去除标点，用python str自带的translate方法
    """
    if text is None:
        return ''
    return text.translate(str.maketrans(dicts))

def cos_sim(v1:[],v2:[]):
    '''向量相似度'''
    v_len=len(v1)
    dot=0.
    square_v1=0.
    square_v2=0.
    for i in range(v_len):
        dot+=v1[i]*v2[i]
        square_v1+=v1[i]*v1[i]
        square_v2+=v2[i]*v2[i]
    return dot/(math.sqrt(square_v1)*math.sqrt(square_v2))

def convert_softmax_value(v:[]):
    '''向量之softmax转换'''
    sum=0.
    for i in range(len(v)):
        sum+=math.exp(v[i])
    for j in range(len(v)):
            v[j]=math.exp(v[j])/sum
    return v
