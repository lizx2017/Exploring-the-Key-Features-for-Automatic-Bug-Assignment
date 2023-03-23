# -*- coding: utf-8 -*-

import pymysql
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import analysis.classify as classify

allFeatures = ['Environment', 'issue_Summary', 'Description', 'Priority', 'Reporter', 'Fix_version',
                    'Component', 'Comment', 'Developer']

expFeatures1 = {'HBASE':['Reporter', 'Developer', 'Environment', 'Component'],
                'CASSANDRA': ['Reporter', 'Developer','Environment', 'Comment'],
                'LUCENE': ['Reporter', 'Developer', 'Component', 'Fix_version'],
                'PDFBOX': ['Comment', 'Reporter', 'Fix_version', 'Developer'],
                'SHIRO': ['Developer', 'Component', 'Reporter', 'Environment']} # Table 4 key features of projects (remove commenters after assignment)
expFeatures2 = {'HBASE':['Reporter', 'Environment'],
                'CASSANDRA': ['Reporter', 'Fix_version'],
                'LUCENE': ['Reporter', 'Fix_version', 'Environment'],
                'PDFBOX': ['Fix_version', 'Reporter'],
                'SHIRO': ['Reporter', 'Component']} # Table 5 key features of projects (self-assignments)
# expFeatures2 = {'HBASE':['Reporter', 'Fix_version', 'Environment', 'Priority'],
#                 'CASSANDRA': ['Reporter', 'Fix_version', 'Description', 'Component'],
#                 'LUCENE': ['Reporter', 'Fix_version', 'Component'],
#                 'PDFBOX': ['Fix_version', 'Component', 'Reporter'],
#                 'SHIRO': ['Fix_version', 'Environment', 'Component']} # Table 7 key features of projects (no self-assignments)
contFeatures1 = ['issue_Summary', 'Description'] # textual features only
contFeatures2 = ['Reporter', 'Component']
contFeatures3 = ['issue_Summary', 'Description', 'Reporter', 'Priority'] # textual features + nominal features
contFeatures4 = ['issue_Summary', 'Description', 'Component', 'Developer'] # textual features + nominal features
contFeatures5 = ['issue_Summary', 'Description', 'Comment', 'Component', 'Priority', 'Fix_version', 'Reporter'] # textual features + nominal features

for project in ['CASSANDRA', 'HBASE']:
    connector = pymysql.connect(user='root', password='137152', host='localhost', database='bugreportdb')
    sql_dev = "SELECT Assignee, count(*) FROM bugreportdb.issues WHERE (Status='Closed' or Status='RESOLVED') and Project IN (%s) and Assignee!='Unassigned' GROUP BY Assignee" % ','.join(
        ['%s'] * len([project]))
    sql = "SELECT * FROM bugreportdb.issues WHERE (Status='Closed' or Status='RESOLVED') and Project IN (%s) and Assignee!='Unassigned';" % ','.join(
        ['%s'] * len([project]))
    all_developers = pd.read_sql_query(sql_dev, con=connector, params=[project])
    developer = all_developers[all_developers['count(*)'] >= 10].Assignee.values
    dataframe = pd.read_sql_query(sql, con=connector, params=[project])
    dataframe = dataframe[dataframe['Assignee'].isin(developer)]
    dic = {label: idx for idx, label in enumerate(np.unique(dataframe['Assignee']))}
    print('User reflection: ', dic)
    dataframe['Assignee'] = dataframe['Assignee'].map(dic)
    groupby_data_orgianl = dataframe['Assignee'].count()
    print(groupby_data_orgianl)
    lab = dataframe['Assignee'].values
    classes = dataframe['Assignee'].unique()
    lab_bi = label_binarize(lab, classes=classes)
    n_classes = lab_bi.shape[1]

    vecDict = {}
    for f in allFeatures:
        vecDict[f] = classify.vector(dataframe, f, [project], 'TextCNN', 'comment_before_assign')

    features_exp1 = np.hstack([vecDict[feature] for feature in expFeatures1[project]])
    features_exp2 = np.hstack([vecDict[feature] for feature in expFeatures2[project]])
    features_con1 = np.hstack([vecDict[feature] for feature in contFeatures1])
    features_con2 = np.hstack([vecDict[feature] for feature in contFeatures2])
    features_con3 = np.hstack([vecDict[feature] for feature in contFeatures3])
    features_con4 = np.hstack([vecDict[feature] for feature in contFeatures4])
    features_con5 = np.hstack([vecDict[feature] for feature in contFeatures5])

    index = 0
    for featureGroup in [features_exp1, features_exp2, features_con1, features_con2, features_con3, features_con4, features_con5]:
        index += 1
        X_train, X_test, y_train, y_test = train_test_split(featureGroup, lab, test_size=0.20)

        model = DecisionTreeClassifier(random_state=0)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print("Decision Tree + index-%d:%f"%(index, acc))

        model = SVC(kernel='rbf', C=1000, gamma=0.001)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print("SVC + index-%d:%f"%(index, acc))

        model = SGDClassifier()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print("SGDClassifier + index-%d:%f"%(index, acc))

        # model = AdaBoostClassifier()
        # model.fit(X_train, y_train)
        # y_pred = model.predict(X_test)
        # acc = accuracy_score(y_pred, y_test)
        # print("AdaBoostClassifier + index-%d:%f"%(index, acc))
        #
        # model = RandomForestClassifier()
        # model.fit(X_train, y_train)
        # y_pred = model.predict(X_test)
        # acc = accuracy_score(y_pred, y_test)
        # print("RandomForestClassifier + index-%d:%f"%(index, acc))

        # model = GradientBoostingClassifier()
        # model.fit(X_train, y_train)
        # y_pred = model.predict(X_test)
        # acc = accuracy_score(y_pred, y_test)
        # print("GradientBoostingClassifier + index-%d:%f" % (index, acc))
        #
        # model = XGBClassifier(max_depth=5, learning_rate=0.01, n_estimators=2000, colsample_bytree=0.1)
        # model.fit(X_train, y_train)
        # y_pred = model.predict(X_test)
        # acc = accuracy_score(y_pred, y_test)
        # print("XGBClassifier + index-%d:%f" % (index, acc))