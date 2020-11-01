import numpy as np
b = 9
a = 0.0 if b == 1 else 1

print(a)
ch_num=10
STATUS_NUM=4
arr=np.array([[0. for col in range(ch_num)] for row in range(STATUS_NUM)])
print(arr)

a2=[1,2]
b2={1,2}
print(type(a2))
print(type(b2))
print(1 in a2)
print(1 in b2)

def get_word_ch_list(word: str):
    '''
    将词语转换为文字列表
    :param word:
    :return:
    '''
    if word is None or word.strip() == '':
        return []
    res = []
    for ch in word:
        res.append(ch)

    return res


def get_status_list_by_chlist(ch_list: []):
    '''
    根据词语的文字列表获取隐藏状态列表:BMES ->[0,1,2,3]
    :param ch_list:
    :return: BMES ->[0,1,2,3]
    '''
    len_ch = len(ch_list)
    if len_ch == 0:
        return []
    if len_ch == 1:
        return [3]

    res = [-1 for i in range(len_ch)]
    res[0] = 0
    res[-1] = 2
    for j in range(1, len_ch - 1):
        res[j] = 1

    return res


s='哈佛大学'
a_matrix = [[0. for i in range(STATUS_NUM)] for i in range(STATUS_NUM)]
ch_list=get_word_ch_list(s)
article_ch_status_list=get_status_list_by_chlist(ch_list)
article_ch_len = len(article_ch_status_list)
for i in range(0, article_ch_len - 1):
    # 状态转移频次统计
    k_status = article_ch_status_list[i]
    j_status = article_ch_status_list[i + 1]
    a_matrix[k_status][j_status] += 1.
# print(a_matrix)
# BMME  BMES 0123 01:1 11:1 12:1

    # print(j+1)
for ch in ch_list[:-1]:
    print(ch)
import math
print(-(1e+10))

