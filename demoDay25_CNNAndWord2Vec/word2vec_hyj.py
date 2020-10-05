import multiprocessing
from gensim.models import Word2Vec,fasttext
from gensim.models.word2vec import LineSentence
from common import common_util

def train_word2vec_model(input_file_path:str, model_file_path:str):
    '''
    训练模型
    参数解释：
    1.sentences：可以是一个List，对于大语料集，建议使用BrownCorpus,Text8Corpus或·ineSentence构建。
    2.sg： 用于设置训练算法，默认为0，对应CBOW算法；sg=1则采用skip-gram算法。
    3.size：是指输出的词的向量维数，默认为100。大的size需要更多的训练数据,但是效果会更好. 推荐值为几十到几百。
    4.window：为训练的窗口大小，8表示每个词考虑前8个词与后8个词（实际代码中还有一个随机选窗口的过程，窗口大小<=5)，默认值为5。
    5.alpha: 是学习速率
    6.seed：用于随机数发生器。与初始化词向量有关。
    7.min_count: 可以对字典做截断. 词频少于min_count次数的单词会被丢弃掉, 默认值为5。
    8.max_vocab_size: 设置词向量构建期间的RAM限制。如果所有独立单词个数超过这个，则就消除掉其中最不频繁的一个。每一千万个单词需要大约1GB的RAM。设置成None则没有限制。
    9.sample: 表示 采样的阈值，如果一个词在训练样本中出现的频率越大，那么就越会被采样。默认为1e-3，范围是(0,1e-5)
    10.workers:参数控制训练的并行数。
    11.hs: 是否使用HS方法，0表示: Negative Sampling，1表示：Hierarchical Softmax 。默认为0
    12.negative: 如果>0,则会采用negative samping，用于设置多少个noise words
    13.cbow_mean: 如果为0，则采用上下文词向量的和，如果为1（default）则采用均值。只有使用CBOW的时候才起作用。
    14.hashfxn： hash函数来初始化权重。默认使用python的hash函数
    15.iter： 迭代次数，默认为5。
    16.trim_rule： 用于设置词汇表的整理规则，指定那些单词要留下，哪些要被删除。可以设置为None（min_count会被使用）或者一个接受()并返回RU·E_DISCARD,uti·s.RU·E_KEEP或者uti·s.RU·E_DEFAU·T的函数。
    17.sorted_vocab： 如果为1（defau·t），则在分配word index 的时候会先对单词基于频率降序排序。
    18.batch_words：每一批的传递给线程的单词的数量，默认为10000
    :param input_file_path:
    :param model_file_path:
    :return:
    '''
    print('train_word2vec_model start...')
    model = Word2Vec(LineSentence(input_file_path),
                     size=400,  # 词向量长度为400
                     window=5,
                     # sg=1,
                     iter=10,
                     workers=multiprocessing.cpu_count())
    print('转换过程结束！')
    print('开始保存模型...')
    model.save(model_file_path)
    print('train_word2vec_model done')
    # print('鲁迅-v:',model['鲁迅'])
    # print('沙瑞金-高育良-sim:',model.wv.similarity('沙瑞金', '高育良'))
    # print('模型保存结束！')

def train_fasttext_model(input_file_path:str, model_file_path:str):
    '''
    fasttext对于不在词表的词可以计算出词向量
    gensim 中Fasttext 模型架构和Word2Vec的模型架构差几乎一样，
    只不过在模型词的输入部分使用了词的n-gram的特征。
    这里需要讲解一下n-gram特征的含义。
    举个例子，如果原词是一个很长的词：你吃了吗。
    jieba分词结果为["你","吃了"，"吗"]。
    unigram(1-gram)的特征：["你","吃了"，"吗"]
    bigram(2-gram) 的特征: ["你吃了"，"吃了吗"]
    所以大家发现没，n-gram的意思将词中连续的n个词连起来组成一个单独的词。
     如果使用unigram和bigram的特征，词的特征就会变成：
     ["你","吃了"，"吗"，"你吃了"，"吃了吗"]这么一长串。
     使用n-gram的词向量使得Fast-text模型可以很好的解决未登录词（OOV——out-of-vocabulary）的问题
    :param input_file_path:
    :param model_file_path:
    :return:
    '''
    print('train_fasttext_model start ...')
    model1 = fasttext.FastText(LineSentence(input_file_path),
                               size=400,
                               window=5,
                               # sg=1,
                               iter=10,
                               workers=multiprocessing.cpu_count())
    model1.save(model_file_path)
    print('train_fasttext_model done')

def load_model(model_path):
    return Word2Vec.load(model_path)
def get_different(words:[],model:Word2Vec):
    '''
    找出不同类的词，这里给出了人物分类题：
    print model.wv.doesnt_match(u"沙瑞金 高育良 李达康 刘庆祝".split())
　　word2vec也完成的很好，输出为"刘庆祝"。
    :param words:
    :param model:
    :return:
    '''
    return model.wv.doesnt_match(words)

def get_sim(w1,w2,model:Word2Vec):
    '''
    看两个词向量的相近程度
    :param w1:
    :param w2:
    :param model:
    :return:
    '''
    return model.wv.similarity(w1, w2)

def topn(model:Word2Vec,word,n=10):
    '''
    找出某一个词向量最相近的词集合
    :param model:
    :param word:
    :param n:
    :return:
    '''
    return model.wv.similar_by_word(word, topn=n)

def update_model(model_path,text_path):
    model = Word2Vec.load(model_path)  # 加载旧模型
    model.build_vocab(LineSentence(text_path), update=True)  # 更新词汇表
    model.train(LineSentence(text_path), total_examples=model.corpus_count, epochs=model.iter)  # epoch=iter语料库的迭代次数；（默认为5）  total_examples:句子数。
    model.save(model_path)
    print('update_model done!')

if __name__ == "__main__":
    print('主程序开始执行...')
    seconds=common_util.get_timestamp()
    file_pre='G:\\bigdata\\badou\\00-data\word2vec\\'
    ''' 1 xml转txt'''
    # xml_path=file_pre+'zhwiki-latest-pages-articles.xml.bz2'
    # txt_file_path = file_pre+'wiki.cn.txt'
    # common_util.wikixml2txt(xml_path,txt_file_path)
    # print('1-seconds:',common_util.get_timestamp(-seconds))

    ''' 2 txt繁体转简体'''
    simple_file_path = file_pre+'wiki.cn.simple.txt'
    # common_util.tradition2simple(txt_file_path,simple_file_path)
    # print('2-seconds:',common_util.get_timestamp(-seconds))

    ''' 3 jieba分词'''
    # jieba_file_path =file_pre+ 'wiki.cn.simple.separate.txt'
    # common_util.jieba_cut_by_file(simple_file_path,jieba_file_path)
    # print('3-seconds:',common_util.get_timestamp(-seconds))

    ''' 4 去除非中文词'''
    wiki_file_path =file_pre+ 'wiki.txt'
    # common_util.zh_trim_by_file(jieba_file_path,wiki_file_path,only_zh=False)
    # print('4-seconds:',common_util.get_timestamp(-seconds))
    #
    ''' 5 开始训练'''
    input_file_path =wiki_file_path
    model_file_path = file_pre+'out\\cbow\\wiki_cbow_10.model'
    model_file_path0 = file_pre+'out\\wiki.model'
    model_file_path2 = file_pre+'out\\cbow\\fast_cbow_10.model'
    model_file_path2_2 = file_pre+'out\\fast.model'
    # print('word2vec train start...')
    # train_word2vec_model(input_file_path,model_file_path)
    # dt1=common_util.get_timestamp()
    # print('5-1-seconds:',common_util.get_timestamp(-seconds))
    # #
    # # print('fast train start...')
    # train_fasttext_model(input_file_path,model_file_path2)
    # print('5-2-seconds:',common_util.get_timestamp(-dt1))

    #6 加载模型
    # model=load_model(model_file_path)
    # model0=load_model(model_file_path0)
    # model2_2=load_model(model_file_path2_2)

    model2=load_model(model_file_path2)
    v1=model2['窦俊']
    print(len(v1),v1)
    # print('----------------------------------')
    # v2=model2.wv['沙瑞金']
    # print(len(v2),v2)
    # v1 v2相同
    # [ -1.72133058e-01  -2.89707989e-01   7.29604065e-01  -5.64783990e-01
    #    1.14486955e-01   3.15245867e-01  -5.68415038e-02  -5.32769442e-01
    #   -7.61959195e-01  -5.72815314e-02   2.09792256e-02  -2.42800310e-01
    #   -7.67124891e-01   1.73165828e-01   3.10316861e-01  -7.49750793e-01
    #   -1.22060573e+00  -1.51866174e+00   2.91436553e-01  -1.57104820e-01
    #    1.96773484e-01   2.39653707e-01   5.20673633e-01   1.78929135e-01
    #   -4.16019052e-01  -7.82062829e-01  -1.83408052e-01  -3.82736981e-01
    #    6.97701946e-02   4.13030326e-01   1.49390090e+00  -1.43545240e-01
    #   -3.23600471e-01   1.46006152e-01  -5.46112418e-01  -9.29802537e-01
    #   -1.92562670e-01   5.25998235e-01  -2.17936710e-01   8.31266958e-03
    #    4.41225290e-01   7.07544684e-01  -1.46203712e-01  -1.05539933e-01
    #    1.00131297e+00   5.46643324e-02  -4.71193850e-01  -2.96569943e-01
    #    5.74219048e-01   1.59659714e-01  -1.87975496e-01  -2.29177520e-01
    #    5.14405549e-01  -7.54959226e-01  -1.20039627e-01   9.71868858e-02
    #   -6.61222219e-01   5.62400043e-01   1.14461708e+00  -8.89693573e-02
    #   -2.06448153e-01   3.53185594e-01  -6.21726394e-01   3.21663380e-01
    #   -1.73530221e-01  -5.18813014e-01  -5.90506792e-01   6.60529733e-01
    #    1.56364851e-02  -7.63624609e-01  -3.58508006e-02  -6.18477285e-01
    #    3.64380211e-01   1.32444054e-01  -4.20839116e-02  -8.53599757e-02
    #    2.75499336e-02   2.94424713e-01  -1.00650787e-01   4.05865610e-01
    #   -3.93781841e-01   3.19147974e-01  -3.44920814e-01  -4.70934153e-01
    #   -3.98188561e-01  -3.68424272e-03   4.86545265e-01   4.34117645e-01
    #    5.84323034e-02  -5.02555192e-01   2.08160326e-01  -2.48909332e-02
    #   -6.74101532e-01   9.79453921e-01  -3.26065838e-01  -2.80905336e-01
    #    3.46054763e-01  -8.85416508e-01  -4.73689079e-01   2.81810075e-01
    #    9.00859833e-02   1.93284631e-01  -7.62110591e-01   8.60868275e-01
    #   -9.63127434e-01   1.20749697e-01  -4.61284161e-01  -7.95888066e-01
    #    1.22012162e+00   1.41660601e-01   4.59264457e-01   4.48091984e-01
    #   -8.59733462e-01  -7.93453217e-01  -2.95560360e-01   1.01128638e-01
    #    1.02975652e-01  -6.41394973e-01  -1.55901223e-01   2.70489872e-01
    #   -2.95844734e-01  -7.44185388e-01  -6.91596806e-01   9.95801166e-02
    #   -5.74901998e-01   4.26604524e-02   2.97979295e-01   2.57844627e-01
    #   -4.14656162e-01  -2.57145375e-01  -4.27569211e-01   3.30390990e-01
    #   -3.19222957e-01  -8.64879727e-01   4.31176841e-01   7.32553005e-01
    #   -3.10743541e-01  -6.15645766e-01   5.44368804e-01  -3.25411201e-01
    #   -2.55431205e-01   2.21089907e-02   2.25195840e-01   2.24997237e-01
    #    3.25701624e-01   1.16750240e-01   2.13827729e-01  -2.85654932e-01
    #    3.88242185e-01   3.69857430e-01  -1.35234284e+00   6.16609395e-01
    #    2.31901594e-02  -5.52156381e-02   2.73417056e-01   9.49718595e-01
    #   -2.02954769e-01  -5.35741225e-02   6.68871701e-02  -1.86964974e-01
    #   -5.33125818e-01   1.15422748e-01  -2.58011937e-01   1.67124644e-01
    #   -6.00208640e-01   3.80003095e-01   2.00638503e-01  -3.80912758e-02
    #    4.71508466e-02   2.67613351e-01   3.59475344e-01  -7.49056876e-01
    #    5.86189330e-01  -1.20596113e-02   1.62587658e-01   3.27195600e-02
    #   -4.68773067e-01   1.32847698e-02  -1.83269948e-01  -4.30688858e-01
    #   -5.54865539e-01   7.38884151e-01   8.95366967e-01  -5.00924170e-01
    #    6.98228180e-01   5.11295080e-01  -2.24200562e-01   6.48843646e-01
    #    2.43138358e-01   4.52312708e-01  -1.22292265e-01  -3.50012749e-01
    #   -5.79589367e-01  -5.91562629e-01  -3.60043421e-02  -1.90905884e-01
    #    1.62624381e-02  -2.36078113e-01  -3.86643291e-01  -6.03297353e-01
    #    1.47975892e-01   5.90647101e-01  -5.77434525e-02   6.62338585e-02
    #   -2.04403643e-02  -3.97239625e-01   3.12873036e-01   6.62234202e-02
    #    2.89873660e-01  -1.88464552e-01  -4.92045581e-01  -2.77896188e-02
    #   -5.56416154e-01   4.18490350e-01   6.55541182e-01  -2.53926486e-01
    #    2.25597784e-01  -6.44823372e-01  -6.67106032e-01   5.51570833e-01
    #   -1.64208531e-01  -1.41971481e+00  -2.33746976e-01   3.47627640e-01
    #   -9.13445950e-01   7.59486556e-02   3.26449037e-01   3.24443281e-01
    #   -1.86890393e-01  -2.74662673e-01   6.31122947e-01   1.00626218e+00
    #   -3.29955280e-01   3.73955667e-01   5.26587307e-01   4.38340515e-01
    #   -1.78104788e-01  -8.85691643e-02  -2.45946646e-01   2.37474084e-01
    #    1.00658250e+00   3.01288098e-01   3.19098145e-01  -2.96427518e-01
    #    6.03283286e-01  -7.35896587e-01  -5.61554611e-01  -8.46643001e-02
    #   -6.57460070e-04  -3.02343816e-01  -3.17138284e-02   4.49867211e-02
    #   -2.56454289e-01   2.77345031e-01  -4.14714664e-01  -2.73814440e-01
    #   -9.19521376e-02  -4.79652509e-02   3.08313817e-01   1.94005623e-01
    #    7.20092297e-01  -5.57759166e-01  -8.30331504e-01   1.51470438e-01
    #    2.51473814e-01   7.07491875e-01   8.04918468e-01  -6.38527572e-01
    #   -2.92259037e-01  -2.48942345e-01  -6.41277671e-01   2.38756776e-01
    #   -8.75963047e-02  -6.51730180e-01  -5.75433612e-01  -7.21461296e-01
    #   -5.68337262e-01   2.23693967e-01  -1.78926557e-01   1.94314107e-01
    #    1.95978470e-02   1.88326865e-01   7.60731578e-01  -8.14616159e-02
    #   -1.57939587e-02  -4.25615400e-01  -4.15136486e-01   2.36666352e-01
    #    1.22890353e+00   6.48019433e-01  -1.95351854e-01   2.37311408e-01
    #   -8.51468861e-01  -4.78730440e-01  -1.76836520e-01  -1.16642915e-01
    #   -5.04887104e-02  -4.05907243e-01   2.45610088e-01   9.67078030e-01
    #   -3.39725792e-01  -2.86777794e-01  -3.02367099e-02   6.59284890e-01
    #    2.85781026e-01  -8.74908328e-01  -5.49187064e-01   2.48326227e-01
    #   -3.97617906e-01   8.85500312e-01   2.73302138e-01  -7.36596584e-02
    #   -9.33793902e-01   4.98861223e-01   1.24574695e-02   4.01235998e-01
    #   -3.11256379e-01  -2.44661272e-02  -3.16916347e-01  -4.45768535e-01
    #    1.14952803e-01  -1.03734368e-02   2.29083270e-01  -1.26633257e-01
    #    1.91966623e-01  -4.53521073e-01  -5.56899667e-01  -1.27122954e-01
    #    9.11732540e-02   8.41087222e-01   5.62169135e-01   6.59286499e-01
    #    4.10769098e-02   5.02836168e-01  -2.51709878e-01   4.88042265e-01
    #   -7.28186592e-02   2.67158691e-02   5.90190113e-01  -2.52038836e-01
    #    1.30719244e-02  -6.18753657e-02   5.75356707e-02   1.94022000e-01
    #    1.96965456e-01   3.25745881e-01  -4.77178961e-01   6.97382152e-01
    #   -1.54426605e-01   3.44288737e-01   6.71280548e-02  -4.51990187e-01
    #   -2.74276912e-01   3.99423867e-01  -4.80859905e-01   1.81795090e-01
    #    3.59895468e-01   1.46771073e-01   3.44799347e-02   9.89435792e-01
    #    5.96315265e-01  -1.42507255e-02   4.88140061e-02   4.73892927e-01
    #   -7.62973845e-01   3.29732627e-01  -1.95298150e-01  -3.30947642e-03
    #   -5.62770069e-01  -2.23099947e-01  -3.08178961e-01  -1.40348643e-01
    #   -1.56101659e-01   5.00916481e-01   1.39807403e-01  -3.25785130e-01
    #    5.21632254e-01  -1.82620674e-01  -8.11801136e-01  -3.90663028e-01
    #   -3.12689275e-01   6.20056808e-01  -1.94277585e-01   9.15646553e-01
    #   -4.93312269e-01  -7.09020257e-01   5.49629033e-02   6.04303181e-01
    #    3.20619106e-01   9.63898972e-02  -1.47757441e-01   3.93305868e-01
    #    3.10342908e-02  -9.87932384e-01  -2.09300071e-01  -5.89717999e-02
    #    5.96839607e-01  -5.59289813e-01  -8.90867710e-01   3.99349406e-02]

    # print(topn(model2,'沙瑞金'))
    # print(topn(model,'沙瑞金'))
    # [('周作人', 0.650015115737915), ('茅盾', 0.5930907726287842), ('胡适', 0.5788863897323608), ('郭沫若', 0.5637531876564026), ('蔡元培', 0.5584905743598938), ('郁达夫', 0.5517114400863647), ('闻一多', 0.5349352955818176), ('陈独秀', 0.5348148345947266), ('朱自清', 0.5271354913711548), ('老舍', 0.5149217247962952)]

    # 7 验证auc
    # print(model.wv.accuracy(questions=file_pre+'questions-words.txt'))
    # print(model2_2.wv.accuracy(questions=file_pre+'questions-words.txt'))
    # print(common_util.get_timestamp(-seconds))
    # dt7=common_util.get_timestamp()
    # print('------------------------------------------------------')
    # print(model2.wv.accuracy(questions=file_pre+'questions-words.txt'))
    # print(dt7)
    # print(common_util.get_timestamp())
    # print(common_util.get_timestamp(-dt7))
    # path='G:\\bigdata\\badou\\00-data\\'
    # update_model(model_file_path2,path+'word2vec\\test\\news_merge.txt')
    print('主程序执行结束！')