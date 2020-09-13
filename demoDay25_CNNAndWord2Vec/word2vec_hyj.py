import multiprocessing
from gensim.models import Word2Vec,fasttext
from gensim.models.word2vec import LineSentence
from common import common_util
import logging

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
    model = Word2Vec(LineSentence(input_file_path),
                     size=400,  # 词向量长度为400
                     window=5,
                     min_count=5,
                     # sg=1,
                     # iter=20,
                     workers=multiprocessing.cpu_count())
    print('转换过程结束！')
    print('开始保存模型...')
    model.save(model_file_path)
    print('鲁迅-v:',model['鲁迅'])
    print('沙瑞金-高育良-sim:',model.wv.similarity('沙瑞金', '高育良'))
    print('模型保存结束！')

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
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    model1 = fasttext.FastText(LineSentence(input_file_path),size=400,window=5)
    model1.save(model_file_path)


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

if __name__ == "__main__":
    print('主程序开始执行...')
    seconds=common_util.get_timestamp()
    file_pre='F:\\八斗学院\\视频\\14期正式课\\00-data\\word2vec\\'
    ''' 1 xml转txt'''
    # xml_path=file_pre+'zhwiki-latest-pages-articles.xml.bz2'
    # txt_file_path = file_pre+'wiki.cn.txt'
    # # common_util.wikixml2txt(xml_path,txt_file_path)
    # # print('1-seconds:',common_util.get_timestamp(-seconds))
    #
    # ''' 2 txt繁体转简体'''
    # simple_file_path = file_pre+'wiki.cn.simple.txt'
    # common_util.tradition2simple(txt_file_path,simple_file_path)
    # print('2-seconds:',common_util.get_timestamp(-seconds))
    # #
    # # ''' 3 jieba分词'''
    # jieba_file_path =file_pre+ 'wiki.cn.simple.separate.txt'
    # common_util.jieba_cut_by_file(simple_file_path,jieba_file_path)
    # print('3-seconds:',common_util.get_timestamp(-seconds))
    #
    # ''' 4 去除非中文词'''
    wiki_file_path =file_pre+ 'wiki.txt'
    # common_util.zh_trim_by_file(jieba_file_path,wiki_file_path)
    # print('4-seconds:',common_util.get_timestamp(-seconds))
    #
    # ''' 5 开始训练'''
    input_file_path =wiki_file_path
    model_file_path = file_pre+'wiki.model'

    # print('转换过程开始...')
    # train_word2vec_model(input_file_path,model_file_path)
    # print('5-seconds:',common_util.get_timestamp(-seconds))
    model=load_model(model_file_path)
    print(model.accuracy(questions='F:\\八斗学院\视频\\14期正式课\\00-data\\word2vec\\qu.txt'))
    # print(topn(model,'鲁迅'))

    print('主程序执行结束！')