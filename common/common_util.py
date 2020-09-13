import os
import re
from string import punctuation
import math
from gensim.corpora import WikiCorpus
import zhconv
import jieba
from datetime import datetime
import logging

# 标点符号集合
punc = punctuation + r'··.,;\\《》？！“”‘’@#￥%…&×（）——+【】{};；●，。&～、|:：'
# 标点符号集合包括字母
punc2=punc + r'\sA-Za-z～()（）【】%*#+-\.\\\/:=：__,，。、;；“”""''’‘？?！!<《》>^&{}|=……'
dicts = {i: '' for i in punc}
dicts2 = {i: '' for i in punc2}
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def get_timestamp(seconds=0.,print_f=True):
    '''
    获取当前时间
    :return: 返回秒值
    '''
    #创建一个datetime类对象
    dt=datetime.now()
    if print_f:
        print_time(dt)
    return dt.timestamp()+seconds

def print_time(dt=datetime.now()):
    '''
    打印当前时间
    :param dt:
    :return:
    '''
    print('时间：' , dt.strftime( '%Y-%m-%d %H:%M:%S %f' ))

def zh_trim_by_file(input_file_path:str, output_file_path:str,only_zh=True):
    '''
    根据文件路径去除非中文词
    :param input_file_path:
    :param output_file_path:
    :return:
    '''
    # input_file_name = 'wiki.cn.simple.separate.txt'
    # output_file_name = 'wiki.txt'
    input_file = open(input_file_path, 'r', encoding='utf-8')
    output_file = open(output_file_path, 'w', encoding='utf-8')
    print('zh_trim_by_file start...')
    lines = input_file.readlines()
    count = 1
    # ^[\u4e00-\u9fa5_a-zA-Z0-9]+$
    if only_zh:
        cn_reg = '^[\u4e00-\u9fa5]+$'
    else:
        cn_reg='^[\u4e00-\u9fa5a-zA-Z]+$'
    for line in lines:
        line_list = line.split('\n')[0].split(' ')
        line_list_new = []
        for word in line_list:
            if re.search(cn_reg, word):
                line_list_new.append(word)
        print(line_list_new)
        output_file.write(' '.join(line_list_new) + '\n')
        count += 1
        if count % 10000 == 0:
            print('目前已分词%d条数据' % count)
    print('zh_trim_by_file done！')




def jieba_cut_by_file(input_file_path:str, output_file_path:str):
    '''
    根据文件路径进行jieba分词
    :param input_file_path:
    :param output_file_path:
    :return:
    '''
    # input_file_path = 'wiki.cn.simple.txt'
    # output_file_path = 'wiki.cn.simple.separate.txt'
    input_file = open(input_file_path, 'r', encoding='utf-8')
    output_file = open(output_file_path, 'w', encoding='utf-8')
    print('jieba_cut_by_file start...')
    lines = input_file.readlines()
    count = 1
    for line in lines:
        # jieba分词的结果是一个list，需要拼接，但是jieba把空格回车都当成一个字符处理
        output_file.write(' '.join(jieba.cut(line.split('\n')[0].replace(' ', ''))) + '\n')
        count += 1
        if count % 10000 == 0:
            print('目前已分词%d条数据' % count)
    print('jieba_cut_by_file done！')




def tradition2simple(input_file_path:str, output_file_path:str):
    '''
    繁体转简体
    :param input_file_path:
    :param output_file_path:
    :return:
    '''
    print('tradition2simple start...')
    # input_file_name = 'wiki.cn.txt'
    # output_file_name = 'wiki.cn.simple.txt'
    input_file = open(input_file_path, 'r', encoding='utf-8')
    output_file = open(output_file_path, 'w', encoding='utf-8')
    print('read tradition start...')
    lines = input_file.readlines()
    print('read tradition done')
    print('tradition convert start...')
    count = 1
    for line in lines:
        output_file.write(zhconv.convert(line, 'zh-hans'))
        count += 1
        if count % 10000 == 0:
            print('tradition convert changed:%d count' % count)
    output_file.close()
    print('tradition2simple done')




def wikixml2txt(input_file_path:str, output_file_path:str):
    '''
    将从网络上下载的xml格式的wiki百科训练语料转为txt格式
    :param input_file_path:
    :param output_file_path:
    :return:
    '''
    print('wikixml2txt start...')
    # input_file_name = 'zhwiki-latest-pages-articles.xml.bz2'
    # output_file_name = 'wiki.cn.txt'
    print('read start...')
    input_file = WikiCorpus(input_file_path, lemmatize=False, dictionary={})
    print('read done')
    output_file = open(output_file_path, 'w', encoding="utf-8")
    print('deal start...')
    count = 0
    for text in input_file.get_texts():
        output_file.write(' '.join(text) + '\n')
        count = count + 1
        if count % 10000 == 0:
            print('目前已处理%d条数据' % count)
    output_file.close()
    print('wikixml2txt:deal done!')


# 根据文件路径（相对或绝对）创建目录
def mkdirs(file):
    dir = os.path.dirname(os.path.abspath(file))
    print(dir)
    if not os.path.exists(dir):
        os.makedirs(dir)


def trim_with_space_flag(text, space_flag=False,trim_abc=False):
    """
    去除标点，用空格替换
    :param trim_abc 是否去除字母 默认否
    """
    if text is None:
        return ''
    replace = " " if space_flag == True else ""
    if trim_abc:
        return re.sub(r"[{}]+".format(punc2), '', text)
    return re.sub(r"[{}]+".format(punc), replace, text)


def trim_with_str_translate(text):
    """
    去除标点，用python str自带的translate方法
    :param trim_abc 是否去除字母 默认否
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

def log(msg,level=logging.INFO):
    logging.log(level,msg=msg)

# if __name__ == '__main__':
#
#     log(msg='主程序开始执行...')
#     cn_reg='^[\u4e00-\u9fa5a-zA-Z]+$'
#     print(re.search(cn_reg,' word'))