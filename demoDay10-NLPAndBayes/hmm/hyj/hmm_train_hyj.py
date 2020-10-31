'''
隐马尔可夫模型
三个参数θ的计算：
假设隐藏状态共M个
初始状态概率（M）= 所有文章第一个字为k状态的频次 / 所有文章第一个字的状态的频次
状态转移概率（M*M）= 所有文章l状态到k状态的频次 / 所有文章l状态的频次
发射概率（M*N，N表示句子的长度）= 所有文章k状态下观测值“我”的频次 / 所有文章k状态的频次

训练：样本为allfiles.txt，该文件一行是一篇文章，且每篇文章已经分词
即在已知每篇文章的分词序列后，计算θ，然后根据θ和观测序列O（如一个未分词的句子），计算最佳的隐藏序列S，
知道S后即可分词

θ的计算：
1 遍历每篇文章，计算每篇文章各个字的隐藏状态S（S in {BMES}）
2 统计各个初始状态的频次，某状态的频次，某状态到某状态的频次，某状态到某个文字的频次
3 根据公式计算3种概率
4 结果存储到磁盘
'''

import math

'''读取数据'''
pre = 'F:\\八斗学院\\视频\\14期正式课\\00-data\\'
train_file = pre + 'allfiles.txt'
# 隐藏状态数量
STATUS_NUM = 4
min_prob = math.log(1e-10)


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


if __name__ == '__main__':
    st = '哈'
    ch_list = get_word_ch_list(st)
    slist = get_status_list_by_chlist(ch_list)
    print(ch_list)
    print(slist)


def train_theta():
    # 初始状态概率 M个
    start_prob = [0. for i in range(STATUS_NUM)]
    # 状态转移概率矩阵 M*M
    a_matrix = [[0. for i in range(STATUS_NUM)] for i in range(STATUS_NUM)]
    # 发射概率矩阵 M*N，N表示某状态下的所有文字种数，由于N不确定，所以用dict存储(key=文字,value=概率)
    b_matrix = [dict() for i in range(STATUS_NUM)]

    with open(train_file, mode='r', encoding='utf-8') as f:
        # 一行是一篇文章
        line = f.readline()
        # 文件读完为止
        while line:
            # 每篇文章的所有单词 split()默认分隔符为空格
            words = line.split()
            if len(words) == 0:
                continue

            # 统计初始状态数量
            start_word = words[0]
            start_ch_list = get_word_ch_list(start_word)
            start_status_list = get_status_list_by_chlist(start_ch_list)
            start_prob[start_status_list[0]] += 1.

            # 一篇文章相当于一个观测序列和一个隐藏序列，文章和文章之间没有关系，所以θ不会跨文章统计
            # 每篇文章的隐藏状态序列S
            article_ch_status_list = []
            # 每篇文章的文字序列O
            article_ch_list = []
            for word in words:
                # 词语的文字序列
                ch_list = get_word_ch_list(word)
                # 词语的状态序列
                status_list = get_status_list_by_chlist(ch_list)
                if len(ch_list) == 0 or len(status_list) == 0:
                    continue

                # 将词语序列转换为文章序列
                article_ch_list.extend(ch_list)
                article_ch_status_list.extend(status_list)

            article_ch_len = len(article_ch_status_list)
            for i in range(0, article_ch_len - 1):
                # 状态转移频次统计
                k_status = article_ch_status_list[i]
                j_status = article_ch_status_list[i + 1]
                a_matrix[k_status][j_status] += 1.

                # 发射频次统计
                count_b(b_matrix, ch_list, i, status_list)
            # 由于i的最大长度减了1，故还要统计每篇文章最后一个字的发射频次
            count_b(b_matrix, ch_list, i+1, status_list)

def count_b(b_matrix, ch_list, i, status_list):
    '''
    发射频次统计
    :param b_matrix:
    :param ch_list:
    :param i:
    :param status_list:
    :return:
    '''
    # 下标为i的文字
    ch_i = ch_list[i]
    # 下标为i的状态
    status_i = status_list[i]
    # 如果某状态下某文字为空，则初始化计数为0
    if b_matrix[status_i].get(ch_i, -1) == -1:
        b_matrix[status_i][ch_i] = 0.
    b_matrix[status_i][ch_i] += 1
