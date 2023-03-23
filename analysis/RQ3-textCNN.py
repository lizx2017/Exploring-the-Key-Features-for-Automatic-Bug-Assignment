import re
import pymysql
import pickle

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from keras_preprocessing.text import Tokenizer
from keras.models import Model, Sequential
from keras.layers import Dense, Flatten, Dropout, MaxPool1D, Conv1D, Embedding, Input
from keras.layers.merging import concatenate
from keras_preprocessing.sequence import pad_sequences
from keras.metrics import top_k_categorical_accuracy

def getData(project):
    db = pymysql.connect(user='root', password='137152', host='localhost', database='bugreportdb')
    cursor = db.cursor()
    # query = "SELECT issueID, Assignee, issue_summary, Description FROM issues AS TableA JOIN (SELECT Assignee, COUNT(*) AS Freq FROM issues WHERE Project='HBASE' GROUP BY Assignee HAVING Freq>=10 ORDER BY Freq DESC) AS TableB ON TableA.Project='HBASE' AND (TableA.Status='Closed' or TableA.Status='RESOLVED') AND TableA.Assignee = TableB.Assignee AND TableA.Assignee!='Unassigned';"
    query = "SELECT issueID, Assignee, issue_summary, Description FROM issues WHERE Project='HBASE' AND (Status='Closed' or Status='RESOLVED') AND Assignee!='Unassigned' AND Self_assign!='True';"
    cursor.execute(query)
    results = cursor.fetchall()
    cursor.close()
    db.close()

    Issues = dict()
    for result in results:
        Issues[result[0]] = dict()
        Issues[result[0]]['assignee'] = result[1]
        Issues[result[0]]['summary'] = result[2]
        Issues[result[0]]['description'] = result[3]
    return Issues

def trainTextCNNModel(project, saveFeature=True):
    Issues = getData(project)
    IDs = list(Issues.keys())

    Assignees = []
    for id in IDs:
        Assignees.append(Issues[id]['assignee'])

    Summaries = []
    for id in IDs:
        summary = Issues[id]['summary'].lower()
        summary = re.sub(r'[!"#$%&()*+,-./:;<=>?@[\]^_`{|}~\t\n]+', ' ', summary)
        temp = []
        for word in summary.split(' '):
            if not word.isdigit():
                temp.append(word)
        summary = ' '.join(temp)
        Summaries.append(summary)

    Descriptions = []
    for id in IDs:
        description = Issues[id]['description'].lower()
        description = re.sub(r'[!"#$%&()*+,-./:;<=>?@[\]^_`{|}~\t\n]+', ' ', description)
        temp = []
        for word in description.split(' '):
            if not word.isdigit():
                temp.append(word)
        description = ' '.join(temp)
        Descriptions.append(description)
    CombinedTexts = [Summaries[i] + ' ' + Descriptions[i] for i in range(len(Summaries))]


    X_train, X_test, Y_train, Y_test = train_test_split(list(range(len(Assignees))), Assignees, test_size=0.2, random_state=42)
    y_labels = list(set(Assignees))
    labelEncoder = LabelEncoder()
    labelEncoder.fit(y_labels)
    num_labels = len(y_labels)
    y_train = to_categorical(labelEncoder.transform(Y_train), num_labels)
    y_test = to_categorical(labelEncoder.transform(Y_test), num_labels)


    word_size = 300
    window = 5
    nb_negative = 16
    min_count = 1
    nb_worker = 4
    nb_epoch = 2
    subsample_t = 1e-5
    nb_sentence_per_batch = 20
    words = {}
    nb_sentence = 0
    total = 0.
    for combinedText in CombinedTexts:
        nb_sentence += 1
        for w in combinedText.split(' '):
            if w not in words:
                words[w] = 0
                words[w] += 1
            total += 1

    words = {i: j for i, j in words.items() if j >= min_count}  # 截断词频
    id2word = {i + 1: j for i, j in enumerate(words)}  # id到词语的映射，0表示UNK
    word2id = {j: i for i, j in id2word.items()}  # 词语到id的映射
    nb_word = len(words) + 1  # 总词数（算上填充符号0）
    subsamples = {i: j / total for i, j in words.items() if j / total > subsample_t}
    subsamples = {i: subsample_t / j + (subsample_t / j) ** 0.5 for i, j in subsamples.items()}  # 这个降采样公式，是按照word2vec的源码来的
    subsamples = {word2id[i]: j for i, j in subsamples.items() if j < 1.}  # 降采样表
    tokenizer = Tokenizer(lower=True, split=" ")
    tokenizer.word_index = word2id



    desc_X_train = [Descriptions[index] for index in X_train]
    desc_X_test = [Descriptions[index] for index in X_test]
    desc_x_train = tokenizer.texts_to_sequences(desc_X_train)
    desc_x_test = tokenizer.texts_to_sequences(desc_X_test)
    desc_x_train = pad_sequences(desc_x_train, maxlen=500)
    desc_x_test = pad_sequences(desc_x_test, maxlen=500)

    summ_X_train = [Summaries[index] for index in X_train]
    summ_X_test = [Summaries[index] for index in X_test]
    summ_x_train = tokenizer.texts_to_sequences(summ_X_train)
    summ_x_test = tokenizer.texts_to_sequences(summ_X_test)
    summ_x_train = pad_sequences(summ_x_train, maxlen=20)
    summ_x_test = pad_sequences(summ_x_test, maxlen=20)

    summ_input = Input(shape=(20,),dtype='float64')
    summ_embedder = Embedding(len(words)+1, 100, input_length=20)
    summ_embed = summ_embedder(summ_input)
    summ_cnn1 = Conv1D(64, 3, padding='same', strides = 1, activation='relu')(summ_embed)
    summ_cnn1 = MaxPool1D(pool_size=4)(summ_cnn1)
    summ_cnn2 = Conv1D(64, 4, padding='same', strides = 1, activation='relu')(summ_embed)
    summ_cnn2 = MaxPool1D(pool_size=4)(summ_cnn2)
    summ_cnn3 = Conv1D(64, 5, padding='same', strides = 1, activation='relu')(summ_embed)
    summ_cnn3 = MaxPool1D(pool_size=4)(summ_cnn3)
    desc_input = Input(shape=(500,),dtype='float64')
    desc_embedder = Embedding(len(words)+1, 300, input_length=500)
    desc_embed = desc_embedder(desc_input)
    desc_cnn1 = Conv1D(64, 3, padding='same', strides = 1, activation='relu')(desc_embed)
    desc_cnn1 = MaxPool1D(pool_size=4)(desc_cnn1)
    desc_cnn2 = Conv1D(64, 4, padding='same', strides = 1, activation='relu')(desc_embed)
    desc_cnn2 = MaxPool1D(pool_size=4)(desc_cnn2)
    desc_cnn3 = Conv1D(64, 5, padding='same', strides = 1, activation='relu')(desc_embed)
    desc_cnn3 = MaxPool1D(pool_size=4)(desc_cnn3)
    summ_cnn = concatenate([summ_cnn1, summ_cnn2, summ_cnn3], axis=-1)
    desc_cnn = concatenate([desc_cnn1, desc_cnn2, desc_cnn3], axis=-1)
    summ_flat = Flatten()(summ_cnn)
    desc_flat = Flatten()(desc_cnn)
    flat = concatenate([summ_flat, desc_flat], axis=-1)
    drop = Dropout(0.2)(flat)
    main_output = Dense(num_labels, activation='softmax')(drop)
    model = Model(inputs=[summ_input, desc_input], outputs=main_output)
    model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
    model.fit([summ_x_train, desc_x_train], y_train, batch_size=32, epochs=15)

    predictResults = model.predict([summ_x_test, desc_x_test])
    print(sum(top_k_categorical_accuracy(y_test, predictResults, k=1))/len(y_test))
    print(sum(top_k_categorical_accuracy(y_test, predictResults, k=2))/len(y_test))
    print(sum(top_k_categorical_accuracy(y_test, predictResults, k=3))/len(y_test))
    print(sum(top_k_categorical_accuracy(y_test, predictResults, k=4))/len(y_test))
    print(sum(top_k_categorical_accuracy(y_test, predictResults, k=5))/len(y_test))

# textFeature_output = model.layers[-2].output
# feature_model = Model(inputs=model.input, outputs = textFeature_output)

# # summary
# main_input = Input(shape=(20,),dtype='float64')
# embedder = Embedding(len(words)+1, 100, input_length=20)
# embed = embedder(main_input)
# cnn1 = Conv1D(64, 3, padding='same', strides = 1, activation='relu')(embed)
# cnn1 = MaxPool1D(pool_size=4)(cnn1)
# cnn2 = Conv1D(64, 4, padding='same', strides = 1, activation='relu')(embed)
# cnn2 = MaxPool1D(pool_size=4)(cnn2)
# cnn3 = Conv1D(64, 5, padding='same', strides = 1, activation='relu')(embed)
# cnn3 = MaxPool1D(pool_size=4)(cnn3)
# cnn = concatenate([cnn1, cnn2, cnn3], axis=-1)
# flat = Flatten()(cnn)
# drop = Dropout(0.2)(flat)
# main_output = Dense(num_labels, activation='softmax')(drop)
# model = Model(inputs=main_input, outputs=main_output)
#
# # description
# main_input = Input(shape=(500,),dtype='float64')
# embedder = Embedding(len(words)+1, 300, input_length=500)
# embed = embedder(main_input)
# cnn1 = Conv1D(64, 3, padding='same', strides = 1, activation='relu')(embed)
# cnn1 = MaxPool1D(pool_size=4)(cnn1)
# cnn2 = Conv1D(64, 4, padding='same', strides = 1, activation='relu')(embed)
# cnn2 = MaxPool1D(pool_size=4)(cnn2)
# cnn3 = Conv1D(64, 5, padding='same', strides = 1, activation='relu')(embed)
# cnn3 = MaxPool1D(pool_size=4)(cnn3)
# cnn = concatenate([cnn1, cnn2, cnn3], axis=-1)
# flat = Flatten()(cnn)
# drop = Dropout(0.2)(flat)
# main_output = Dense(num_labels, activation='softmax')(drop)
# model = Model(inputs=main_input, outputs=main_output)
    if saveFeature:
        textFeature_output = model.layers[-2].output
        feature_model = Model(inputs=model.input, outputs = textFeature_output)
        summ = concatenate([summ_x_train, summ_x_test], axis=-2)
        desc = concatenate([desc_x_train, desc_x_test], axis=-2)
        features = feature_model.predict([summ, desc])
        featureDict = dict()
        for i in range(len(desc_x_train)):
            featureDict[IDs[X_train[i]]] = features[i]
        for i in range(len(desc_x_test)):
            featureDict[IDs[X_test[i]]] = features[i + len(desc_x_train)]
        f = open(project.lower() + '.pkl', 'wb')
        pickle.dump(featureDict, f)
        f.close()

if __name__=='__main__':
    for project in ['PDFBOX', 'SHIRO', 'LUCENE', 'CASSANDRA', 'HBASE']:
        getData(project)
        trainTextCNNModel(project)
