'''
基于文本相似度进行推荐
1 计算每篇文章各词的wordvec=tfidf*word2vec
语料库为2454篇新闻文章
tf=词频/文章的总词数
idf=文章总数/（出现该词的文章数+1）
2 计算文章的向量docvec=sum(wordvec)/len(set(word))
3 计算文本相似度 取top-N
'''