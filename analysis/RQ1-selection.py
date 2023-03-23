# -*- encoding: utf-8 -*-

import numpy as np
import pymysql
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm
from itertools import cycle
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score, accuracy_score, matthews_corrcoef
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import label_binarize, LabelEncoder
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

import tools

def getGlove():
    f = open('../dataset/glove.840B.300d.txt', encoding='utf-8')
    lines = f.readlines()
    f.close()
    gloveDict = {}
    for line in lines:
        items = line.strip('\n').split(' ')
        vec = np.asarray(items[1:], dtype='float32')
        gloveDict[items[0]] = vec
    return gloveDict

def vector(dataframe, feature, project, strategy, filter = 'noComment'):#, gloveDict
    '''
    特征向量化

    :param dataframe: dataframe格式的原始数据
    :param features:
    :return: 返回每个issue的特征向量
    '''

    TEXTUAL = set(['Comment', 'Description', 'issue_Summary'])
    NOMINAL = ('Environment','Developer','Priority','Component','Fix_version','Reporter', 'Project')

    vector = []
    if strategy == 'textCNN':
        f = open('D:/Python_Project/bugAssignment/' + project[0].lower() + '.pkl', 'rb')
        textFeatureDict = pickle.load(f)
        f.close()

    if feature in TEXTUAL:
        if feature == 'Comment':
            labels = dataframe['issueID'].values
            connector = tools.sql_connect(user='root', password='137152', host='localhost', database='bugreportdb')
            if filter == "comment_before_assign":
                comments = tools.sql_query(
                    "SELECT * from comments where Project IN (%s) and compare2assign!='after';" % ','.join(
                        ['%s'] * len(project)), connector=connector, params=project)
            else:
                comments = tools.sql_query(
                    "SELECT * from comments where Project IN (%s);" % ','.join(
                        ['%s'] * len(project)), connector=connector, params=project)
            sentences = []
            for label in labels:
                temp = comments[comments['issueID'] == label]
                sentences.append('\n'.join(temp['Comment']))
            texts = tools.textual_preprocess(sentences)  # , gloveDict.keys()
            tfidf = tools.TFIDF(texts)  # , gloveDict
        elif strategy != 'textCNN':
            sentences = dataframe[feature].values
            texts = tools.textual_preprocess(sentences)#, gloveDict.keys()
            tfidf = tools.TFIDF(texts)  # , gloveDict
        else:
            if feature == 'issue_Summary':
                tfidf = []
                labels = dataframe['issueID'].values
                for label in labels:
                    tfidf.append(textFeatureDict[label][:960])
                tfidf = np.array(tfidf)
            elif feature == 'Description':
                tfidf = []
                labels = dataframe['issueID'].values
                for label in labels:
                    tfidf.append(textFeatureDict[label][960:])
                tfidf = np.array(tfidf)
        print(feature, tfidf.shape)
        return tfidf

    # if vector == []:
    #     vector = tfidf
    # else:
    #     vector = np.hstack((vector, tfidf))

    # 计算类别类型特征的向量
    if feature in NOMINAL:
        if feature == 'Developer':
            labels = dataframe['issueID'].values
            connector = tools.sql_connect(user='root', password='137152', host='localhost', database='bugreportdb')
            if filter == 'comment_before_assign':
                comments = tools.sql_query(
                    "SELECT * from comments where Project IN (%s) and compare2assign!='after';" % ','.join(
                        ['%s'] * len(project)), connector=connector, params=project)
            else:
                comments = tools.sql_query(
                    "SELECT * from comments where Project IN (%s);" % ','.join(
                        ['%s'] * len(project)), connector=connector, params=project)
            nominals = []
            for label in labels:
                temp = comments[comments['issueID'] == label]
                nominals.append(', '.join(list(set(temp['Developer']))))
        else:
            nominals = dataframe[feature].values
        temp, classes = tools.classical_vector(nominals)
        # temp = temp.values
        print(feature, temp.shape)
        if vector == []:
            vector = temp
        else:
            vector = np.hstack((vector, temp))
    return vector

def label(dataframe):
    '''
    提取数据库中Assignee字段，得到类别标签
    :param dataframe:
    :return:
    '''
    label = tools.numerical_vector(dataframe=dataframe, keyword='Assignee')
    return label

def ROC(y_test, y_score, n_classes):
    # 计算每一类的ROC
    if (y_test.shape.__len__()==1):
        y_test = label_binarize(y_test, classes=n_classes)
        y_score = label_binarize(y_score, classes=n_classes)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        # fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        fpr[i], tpr[i], _ = roc_curve(y_test[i], y_score[i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area（方法二）
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    return roc_auc["micro"]

    # Compute macro-average ROC curve and ROC area（方法一）
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in n_classes]))
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in n_classes:
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    # Finally average it and compute AUC
    mean_tpr /= len(n_classes)
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    print(roc_auc['micro'], roc_auc['macro'])

def Classify(Project='All', strategy = 'tfidf', filter = "noComment", algorithm='SVM', metric='AUC'):
    connector = tools.sql_connect(user='root', password='137152', host='localhost', database='bugreportdb')
    if Project=='All':
        params = ['HBASE', 'LUCENE', 'CASSANDRA', 'SHIRO', 'PDFBOX']
    else:
        params = [Project]
    sql_dev = "SELECT Assignee, count(*) FROM bugreportdb.issues WHERE (Status='Closed' or Status='RESOLVED') and Project IN (%s) and Assignee!='Unassigned' GROUP BY Assignee" % ','.join(['%s'] * len(params))
    sql = "SELECT * FROM bugreportdb.issues WHERE (Status='Closed' or Status='RESOLVED') and Project IN (%s) and Assignee!='Unassigned' and Self_assign!='True';" % ','.join(['%s'] * len(params)) #  and Assignee!=Reporter
    all_developers = tools.sql_query(sql_dev, connector=connector, params=params)
    developer = all_developers[all_developers['count(*)'] >= 10].Assignee.values
    dataframe = tools.sql_query(sql, connector=connector, params=params)
    dataframe = dataframe[dataframe['Assignee'].isin(developer)]
    lab = label(dataframe)
    # classes = dataframe['Assignee'].unique()
    # lab = label_binarize(lab, classes=classes)
    # n_classes = lab.shape[1]

    if filter == 'noComment':
        FEATURES = ['Environment','issue_Summary','Description','Priority','Reporter','Fix_version','Component']
    else:
        FEATURES = ['Environment', 'issue_Summary', 'Description', 'Priority', 'Reporter', 'Fix_version',
                    'Component', 'Comment', 'Developer']
    forward = []
    backward = FEATURES.copy()
    SCORE = 0
    UPPER = 0
    vec_dict = {}
    i = 0
    for f in FEATURES:
        vec_dict[f] = vector(dataframe, f, params, strategy, filter)#, gloveDict
    while(set(forward)!=set(backward)):
        score_list = []
        for feature in FEATURES:
            forward_temp = forward.copy()
            forward_temp.append(feature)
            vec = vec_dict[forward_temp[0]]
            for i in range(1, len(forward_temp)):
                vec = np.hstack((vec, vec_dict[forward_temp[i]]))
            total = 0
            for i in range(2):
                x_train, x_test, y_train, y_test = train_test_split(vec, lab, test_size=0.2)
                if algorithm == "SVM":
                    classifier = svm.SVC(kernel='linear', probability=True)
                elif algorithm == 'Decision Tree':
                    classifier = DecisionTreeClassifier()
                elif algorithm == 'SGDClassifier':
                    classifier = SGDClassifier()
                y_score = classifier.fit(x_train, y_train).predict(x_test)
                if metric == "AUC":
                    score = ROC(y_test, y_score, n_classes) # use AUC as metrics
                elif metric == "accuracy":
                    score = accuracy_score(y_test, y_score) # use accuracy as metrics
                elif metric == "mcc":
                    score = matthews_corrcoef(y_test, y_score) # use matthews correlation coefficient as metrics
                print(score)
                total += score
            score_list.append(total/2)
            print(feature, total/2)
        if len(score_list) != 0 and max(score_list)>SCORE:
            SCORE = max(score_list)
            feature = FEATURES[score_list.index(max(score_list))]
            forward.append(feature)
            FEATURES = list(set(FEATURES).difference(set([feature])))
            print('+', feature, '\tforward:', forward)
        score_list = []
        for feature in FEATURES:
            backward_temp = list(set(backward).difference(set([feature])))
            vec = vec_dict[backward_temp[0]]
            for i in range(1, len(backward_temp)):
                vec = np.hstack((vec, vec_dict[backward_temp[i]]))
            total = 0
            for i in range(2):
                x_train, x_test, y_train, y_test = train_test_split(vec, lab, test_size=0.3)
                if algorithm == "SVM":
                    classifier = svm.SVC(kernel='linear', probability=True)
                elif algorithm == 'Decision Tree':
                    classifier = DecisionTreeClassifier()
                elif algorithm == 'SGDClassifier':
                    classifier = SGDClassifier()
                y_score = classifier.fit(x_train, y_train).predict(x_test)
                if metric == "AUC":
                    score = ROC(y_test, y_score, n_classes) # use AUC as metrics
                elif metric == "accuracy":
                    score = accuracy_score(y_test, y_score) # use accuracy as metrics
                elif metric == "mcc":
                    score = matthews_corrcoef(y_test, y_score) # use matthews correlation coefficient as metrics
                print(score)
                total += score
            score_list.append(total / 2)
            print(feature, total / 2)
        if len(score_list) != 0:
            feature = FEATURES[score_list.index(max(score_list))]
            backward = list(set(backward).difference(set([feature])))
            FEATURES = list(set(FEATURES).difference(set([feature])))
            print('-', feature, '\tbackward:', backward)
        i += 1
    print('Final selections:', forward)


if __name__=='__main__':
    for strategy in ['tfidf', 'TextCNN']:
        for project in ['PDFBOX', 'SHIRO', 'LUCENE', 'CASSANDRA', 'HBASE']:
            print(project + " + " + strategy)
            Classify(project, strategy)