# -*- encoding: utf-8 -*-

import numpy as np
import pandas as pd
import math
import string
import pymysql
from string import digits
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from collections import defaultdict, Counter
from nltk.stem import PorterStemmer, SnowballStemmer, LancasterStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing import MultiLabelBinarizer
from scipy.stats import chi2_contingency # 用于卡方检验

def sql_connect(user, password, host, database):
    '''
    链接数据库

    :param user: 用户名/root
    :param password: 密码/137152
    :param host: 主机/localhost
    :param database: 数据库/bugreportdb
    :return: connector
    '''
    return pymysql.connect(user=user, password=password, host=host, database=database)

def sql_query(query, connector, params=None):
    '''
    数据库查询语句

    :param query: 查询语句
    :param connector: 数据库链接connector
    :return: 查询结果
    '''
    return pd.read_sql_query(query, con=connector, params=params)

def classical_vector(features):
    '''
    类别行特征向量化（one-hot）

    :param dataframe: dataframe格式的原始数据
    :param keyword: 列名
    :return: 该列的one-hot向量
    '''

    y = [str(i).split(', ') for i in features]
    model = MultiLabelBinarizer()
    return model.fit_transform(y), model.classes_

def numerical_vector(dataframe, keyword):
    dic = {label: idx for idx, label in enumerate(np.unique(dataframe[keyword]))}
    if keyword == "Assignee":
        print('User reflection: ', dic)
    dataframe[keyword] = dataframe[keyword].map(dic)
    groupby_data_orgianl = dataframe[keyword].count()
    print(groupby_data_orgianl)
    vector = dataframe[keyword].values
    return vector

def remove_stopWord(sentences):
    '''
    给定分词集合，去除集合中的stop-word（常见词a, the）
    :param sentences: 分词集合
    :return: clean_tokens, 分词集合
    '''
    sr = stopwords.words('english')
    clean_token = [[token for token in str(tokens).lower().split() if token not in sr] for tokens in sentences]
    return clean_token

def Stemmer(sentences):
    '''
    提取词干
    :param tokens: 词集
    :return: 词干集
    '''
    stemmer = PorterStemmer()
    stems = [[stemmer.stem(token) for token in sentence] for sentence in sentences]
    return stems

def numFilter(text):
    for ch in text:
        if ch in '0123456789':
            return False
    return True

def textual_preprocess(sentences):

    punc = str.maketrans(string.punctuation, ' '*len(string.punctuation), digits)
    tokens = remove_stopWord(sentences)
    sentences = [' '.join(token) for token in tokens]
    sentences = [sentence.translate(punc) for sentence in sentences]
    tokens = [sentence.split() for sentence in sentences]
    frequency = defaultdict(int)
    for token in tokens:
        for t in token:
            frequency[t] += 1
    texts = [[token for token in text if numFilter(token)] for text in tokens]
    texts = [[token for token in text if frequency[token] > 1] for text in texts]
    return texts

def TFIDF(texts):
    '''
    计算TFIDF
    :param texts: 由[[词]句]构成的二维list
    :return: TFIDF vector
    '''
    vector = []
    descriptions = [' '.join(text) for text in texts]
    vectorizer = CountVectorizer(stop_words='english')
    X = vectorizer.fit_transform(descriptions)
    word = vectorizer.get_feature_names()
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(X)
    tfidf = tfidf.toarray()
    return tfidf

def extractCode(dataframe, feature):
    texts = dataframe[feature].values
    sentences = [[text.split('. ')] for text in texts]

def discriminative_feature(dataframe, feature, label, type):
    labels = dataframe[label].values
    dic_l = np.unique(labels)
    label_count = [labels.count(label) for label in dic_l]
    if type == 'textual':
        sentences = dataframe[feature].values
        items = textual_preprocess(sentences)
        dic = []
        for item in items:
            for word in item:
                if word not in dic:
                    dic.append(word)
    elif type == 'nominal':
        temp = dataframe[feature].values
        items = [[t.split(', ')] for t in temp]
        dic = []
        for item in items:
            for cate in item:
                if cate not in dic:
                    dic.append(cate)
    matrix = []
    for item in dic:
        temp = np.zeros(len(dic_l))
        for i in range(len(items)):
            if item in items[i]:
                temp[dic_l.index(labels[i])] += 1
        matrix.append(temp)
    return CHI(matrix, label_count, dic)

def CHI(matrix, count, dic):
    chi = dict()
    for i in range(len(dic)):
        temp = count - matrix[i]
        chi[dic[i]] = chi2_contingency([matrix[i], temp])[1] # 取值越小越相关
    return chi

def entropy(project, sql):
    '''
    为了计算样本的不均衡度
    :param project: 项目
    :param sql: 查询语句
    :return: 信息熵
    '''
    connector = sql_connect(user='root', password='137152', host='localhost', database='bugreportdb')
    dataframe = sql_query(sql, connector=connector)
    dev_count_dict = dict(zip(dataframe['Assignee'], dataframe['count(*)']))
    entro = 0
    total = sum(dev_count_dict.values())
    count = dataframe.size
    for key in dev_count_dict.keys():
        temp = -dev_count_dict[key]/total * math.log(dev_count_dict[key]/total)
        entro += temp
    print('The entropy of ', project, ' is:\t', entro)
    print(project, ':\t', entro/math.log(count))
    return entro