'''
https://www.kaggle.com/competitions/sentiment-analysis-on-movie-reviews/overview/evaluation
烂番茄电影评论数据集是用于情感分析的电影评论语料库，最初由 Pang 和 Lee [1] 收集。在他们关于情绪树库的工作中，Socher 等人。
[2] 使用 Amazon 的 Mechanical Turk 为语料库中的所有已解析短语创建细粒度标签。
本次比赛提供了一个机会，可以在烂番茄数据集上对您的情绪分析想法进行基准测试。
您被要求在五个值的范围内标记短语：消极、有些消极、中性、有些积极、积极。句子否定、讽刺、简洁、语言歧义等许多障碍使这项任务非常具有挑战性。

提交的内容会根据每个解析短语的分类准确度（正确预测的标签百分比）进行评估。情绪标签是：
0 - 消极
1 - 有点消极
2 - 中性
3 - 有点积极
4 - 积极
'''

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
import os
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer

CV = CountVectorizer(ngram_range=(2, 2))

nltk.download('stopwords')
DATA_ROOT = 'F:\\00-data\\nlp\\sentiment-analysis-on-movie-reviews\\'
for dirname, _, filenames in os.walk(DATA_ROOT):
    for filename in filenames:
        print(os.path.join(dirname, filename))

train_data = pd.read_csv(DATA_ROOT + 'train.tsv.zip', sep='\t')
test_data = pd.read_csv(DATA_ROOT + 'test.tsv.zip', sep='\t')

NUM_SAMPLES = train_data.shape[0]
train_data = train_data.head(NUM_SAMPLES)
labels = train_data['Sentiment']
# X_train, X_test, y_train, y_test=train_test_split(train_data,labels,random_state=2022,test_size=0.2)
X_train, X_test, y_train = train_data, test_data, labels
print(X_train.shape, X_test.shape)
train_data.head()


def process_phrase(ph):
    # Using Regular Expressions to further process the string
    ph = re.sub("[^a-zA-Z?!.;:]",  # The pattern to search for
                " ",  # The pattern to replace it with
                ph)  # The text to search

    # We will convert the string to lowercase letter and divide them into words
    words = ph.lower().split()

    # Searching a set is much faster than searching list, so we will convert the stop words into a set
    stops = set(stopwords.words("english"))

    # We now remove the stop words or the unimportant words and retain only meaningful ones
    mean_words = [w for w in words if not w in stops]
    return " ".join(mean_words)


# Processing Each Phrase
X_train['Phrase'] = [process_phrase(p) for p in X_train['Phrase']]
X_test['Phrase'] = [process_phrase(p) for p in X_test['Phrase']]

# Implementing BOW Model
vectorizer = CountVectorizer(analyzer='word',
                             tokenizer=None,
                             preprocessor=None,
                             stop_words=None,
                             max_features=5000, ngram_range=(1, 2))
# 0.616717929001666 ngram_range=(1, 2)
train_data_features = vectorizer.fit_transform(X_train['Phrase'])

print(train_data_features.A)
print(vectorizer.get_feature_names())
print(train_data_features.shape)
'''
1 逻辑回归进行多分类
multinomial表示使用softmax，ovr表示将某类看作正样本 其余为负样本进行二分类，训练m个分类器
'''
log_rot = LogisticRegression(multi_class='multinomial', solver='newton-cg')
log_rot.fit(train_data_features, y_train)
print('w:%s, b:%s' % (log_rot.coef_, log_rot.intercept_))
# Pre-processing Test Data
test_data_features = vectorizer.transform(X_test['Phrase'])
test_data_features = test_data_features.toarray()


def predict(model, y_test, logPre):
    results = model.predict(y_test)
    print(results)
    # y_test=np.array(y_test)
    # print(log_rot.score(test_data_features,y_test))
    # Saving submissions
    output_file = pd.DataFrame(data={'PhraseId': test_data['PhraseId'], 'Sentiment': results})
    output_file.to_csv(logPre + '-mozhu.csv', index=False)
    print('The submission file has been saved successfully')
    return


# predict(log_rot,'lr')

from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier(n_estimators=50,n_jobs=4)
forest.fit(train_data_features, train_data['Sentiment'])
predict(forest, test_data_features, 'forest')


