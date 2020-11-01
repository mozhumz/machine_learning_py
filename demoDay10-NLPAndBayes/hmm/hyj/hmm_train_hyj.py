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
save_file = pre + 'out\\hmm\\hyj_hmm.model'
# 隐藏状态数量
STATUS_NUM = 4
# 频次为0的默认无穷小概率
min_prob = -100000.0


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


def train_theta():
    # 初始状态概率 M个
    start_prob = [0. for i in range(STATUS_NUM)]
    # 状态转移概率矩阵 M*M
    a_matrix = [[0. for i in range(STATUS_NUM)] for i in range(STATUS_NUM)]
    # 发射概率矩阵 M*N，N表示某状态下的所有文字种数，由于N不确定，所以用dict存储(key=文字,value=概率)
    b_matrix = [dict() for i in range(STATUS_NUM)]
    # 初始状态总数
    total_start_status = 0.
    # 每个状态的总频次数
    total_status_dict = dict()
    with open(train_file, mode='r', encoding='utf-8') as f:
        '''
        readline(): 
        1、每次读取下一行文件。 
        2、可将每一行数据分离。 
        3、主要使用场景是当内存不足时，使用readline()可以每次读取一行数据，只需要很少的内存。 
        readlines(): 
        1、一次性读取所有行文件。 
        2、可将每一行数据分离，从代码中可以看出，若需要对每一行数据进行处理，可以对readlines()求得的结果进行遍历。 
        3、若内存不足无法使用此方法。
        '''
        # 一行是一篇文章
        line = f.readline()
        # 文件读完为止
        while line and line.strip() != '':
            print('while line==True')
            # 每篇文章的所有单词 split()默认分隔符为空格
            words = line.strip().split()
            if len(words) == 0:
                return

            # 统计初始状态频次
            start_word = words[0]
            start_ch_list = get_word_ch_list(start_word)
            start_status_list = get_status_list_by_chlist(start_ch_list)
            start_prob[start_status_list[0]] += 1.
            total_start_status += 1

            # 一篇文章相当于一个观测序列和一个隐藏序列，文章和文章之间没有关系，所以θ不会跨文章统计
            # 每篇文章的隐藏状态序列S
            article_ch_status_list = []
            # 每篇文章的文字序列O
            article_ch_list = []
            #  words[:-1]最后一个不是词语，需要过滤
            for word in words[:-1]:
                # 词语的文字序列
                ch_list = get_word_ch_list(word)
                # 词语的状态序列
                status_list = get_status_list_by_chlist(ch_list)
                if len(ch_list) == 0 or len(status_list) == 0:
                    continue

                # 将词语序列转换为文章序列
                article_ch_list.extend(ch_list)
                article_ch_status_list.extend(status_list)

            # 每篇文章的频次统计
            article_ch_len = len(article_ch_status_list)
            for i in range(0, article_ch_len - 1):
                # 状态转移频次统计
                k_status = article_ch_status_list[i]
                j_status = article_ch_status_list[i + 1]
                a_matrix[k_status][j_status] += 1.
                count_status(k_status, total_status_dict)

                # 发射频次统计
                count_b(b_matrix, article_ch_list, i, article_ch_status_list)
            # 由于i的最大长度减了1，故还要统计每篇文章最后一个字的状态频次和发射频次
            count_status(article_ch_status_list[-1], total_status_dict)
            count_b(b_matrix, article_ch_list, i + 1, article_ch_status_list)
            line = f.readline()

    print('count pi:', start_prob)
    print('count A:', a_matrix)
    print('count B:', b_matrix)

    '''计算概率'''
    print('compute  prob...')
    for i in range(STATUS_NUM):
        # 初始状态概率
        # 概率取对数
        prob = start_prob[i] / total_start_status
        if prob > 0:
            start_prob[i] = math.log(prob)
        else:
            # 如果频次为0 则给一个默认的无穷小概率
            start_prob[i] = min_prob

        # 状态转移概率
        for j in range(STATUS_NUM):
            prob = a_matrix[i][j] / total_status_dict[i]
            if prob > 0.:
                a_matrix[i][j] = math.log(prob)
            else:
                a_matrix[i][j] = min_prob

        # 发射概率
        for single, count in b_matrix[i].items():
            prob = count / total_status_dict[i]
            if prob > 0.:
                b_matrix[i][single] = math.log(prob)
            else:
                b_matrix[i][single] = min_prob

    print(' start-prob:', start_prob)
    print('a-prob:', a_matrix)
    print('b-prob:', b_matrix)

    print('save model...')
    with open(save_file, mode='w', encoding='utf-8') as f:
        f.write(str(start_prob) + '\n')
        f.write(str(a_matrix) + '\n')
        f.write(str(b_matrix) + '\n')
    print('save done:', save_file)


def count_status(k_status, total_status_dict):
    if total_status_dict.get(k_status, -1) == -1:
        total_status_dict[k_status] = 0.
    total_status_dict[k_status] += 1.


def viterbi(sentence, sep=' '):
    '''
    给定一个未分词的句子，使用维特比算法，寻找最优的切分序列S
    :param sentence:
    :param sep: 分隔符
    :return: 返回不同的分词方案
    '''
    res = []

    # 加载模型
    # 初始状态概率 M
    PI = None
    # 状态转移概率 M*M
    A = None
    # 发射概率 M*N N表示某状态下的文字种数
    B = None
    with open(save_file, mode='r', encoding='utf-8')as f:
        PI = eval(f.readline())
        A = eval(f.readline())
        B = eval(f.readline())

    # 获取句子的文字列表
    ch_list = get_word_ch_list(sentence)
    sentence_len = len(ch_list)
    # M*句子长度的矩阵，每个元素是一个数组[hmm概率值，上一列中最好的节点状态S]
    s_matrix = [[[0., 0] for j in range(sentence_len)] for i in range(STATUS_NUM)]
    # 计算第一列的hmm概率=PI(S1)*B(O1|S1)
    for i in range(STATUS_NUM):
        if B[i].get(ch_list[0],-1)==-1:
            s_matrix[i][0][0]= PI[i]+min_prob
        else:
            # 由于概率取了对数，相加即相乘
            s_matrix[i][0][0] = PI[i] + B[i][ch_list[0]]
        s_matrix[i][0][1]=i

    # 找到h层的节点i，使得i到h+1层的j节点的hmm值最大
    for ch_j in range(1, sentence_len):
        # 某一行某个文字可能有M种状态
        # 当前列的j
        for j in range(STATUS_NUM):
            max_p = None
            max_status = None
            # 上一列的i
            for i in range(STATUS_NUM):
                # i到j的转移概率
                a_ij = A[i][j]
                # 当前i到j的hmm概率
                cur_p = s_matrix[i][ch_j-1][0] + a_ij
                if max_p is None or max_p < cur_p:
                    max_p = cur_p
                    max_status = i

            # j节点的hmm概率和i节点的状态S
            b_prob=min_prob
            if B[j].get(ch_list[ch_j],-1)!=-1:
                b_prob=B[j][ch_list[ch_j]]

            s_matrix[j][ch_j][0] = max_p +b_prob
            s_matrix[j][ch_j][1] = max_status

    print('s_matrix:',s_matrix)
    # 找到hmm概率最大的S序列
    # 最后一列中，hmm最大的节点
    best_end=None
    # 最后一列中，hmm的最大概率
    best_p=None
    best_end_status=None
    last_node=[]
    for i in range(STATUS_NUM):
        end_i=s_matrix[i][-1]
        last_node.append((end_i[0],end_i[1],i))
        if best_p is None or best_p<end_i[0]:
            best_p=end_i[0]
            best_end_status=i
            # best_end=end_i

    last_node=sorted(last_node,key=lambda x:x[0],reverse=True)
    print(last_node)
    # 从后往前遍历hmm概率矩阵，找出S序列
    s_list=[0 for i in range(sentence_len)]
    s_list[-1]=best_end_status


    for col_i in range(1,sentence_len+1):
        s_list[sentence_len-col_i]=best_end_status
        i_node=s_matrix[best_end_status][sentence_len-col_i]
        best_end_status=i_node[1]
        # best_end=i_node

    print(s_list)
    # 切分句子 BMES 0123
    word=''
    for ch_i in range(sentence_len):
        if s_list[ch_i]==3:
            word=ch_list[ch_i]
            res.append(word)
            word=''
            continue

        if s_list[ch_i]==2:
            word+=ch_list[ch_i]
            res.append(word)
            word=''
            continue

        word+=ch_list[ch_i]

    return res



if __name__ == '__main__':
    # train_theta()
    st = '无边落木萧萧下'
    res=viterbi(st)
    print(res)
    # ch_list = get_word_ch_list(st)
    # slist = get_status_list_by_chlist(ch_list)
    # print(ch_list)
    # print(slist)
