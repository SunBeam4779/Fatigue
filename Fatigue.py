'''
Created on 2018年12月23日

@author: 37434
'''
import numpy as np
import pandas as pd
from sklearn import svm
import time
from builtins import len

PATH = "G:\\CAD\\data.csv"

def read_data(path=PATH):
    data = pd.read_csv(path, encoding='ANSI', delimiter=",")
    data = data.drop([0])
    data_mat = np.array(data)
    data_mat = np.delete(data_mat, 0, axis=1)
    return data_mat

def get_data_and_labels(data, test_ratio):
    datanum = int(len(data)/2)
    tedata_ = []
    trdata_ = []
    telabel_0 = []
    telabel_1 = []
    trlabel_0 = []
    trlabel_1 = []
    telabel_ = []
    trlabel_ = []
    test_set_size = int(datanum*test_ratio)
    data0 = data[:datanum+1, :28]
    label0 = data[:datanum+1, -1]
    data1 = data[datanum+1:, :28]
    label1 = data[datanum+1:, -1]
    tedata0 = data0[:test_set_size]
    telabel0 = label0[:test_set_size]
    tedata1 = data1[:test_set_size]
    telabel1 = label1[:test_set_size]
    trdata0 = data0[test_set_size:]
    trlabel0 = label0[test_set_size:]
    trdata1 = data1[test_set_size:]
    trlabel1 = label1[test_set_size:]
    tedata = np.concatenate((tedata0, tedata1), axis=0)
    trdata = np.concatenate((trdata0, trdata1), axis=0)
    # telabel = np.append(telabel0, telabel1)
    # trlabel = np.append(trlabel0, trlabel1)
    for i in range(len(telabel0)):
        telabel_0.append(telabel0[i])
    for i in range(len(telabel1)):
        telabel_1.append(telabel1[i])
    for i in range(len(trlabel0)):
        trlabel_0.append(trlabel0[i])
    for i in range(len(telabel1)):
        trlabel_1.append(trlabel1[i])
    for i in range(len(tedata)):
        for j in range(len(tedata[i])):
            s = float(tedata[i][j])
            tedata_.append(s)
    for i in range(len(trdata)):
        for j in range(len(trdata[i])):
            k = float(trdata[i][j])
            trdata_.append(k)
    telabel_ = np.concatenate((telabel_0,telabel_1), axis=0)
    trlabel_ = np.concatenate((trlabel_0,trlabel_1), axis=0)
    tedata_ = np.reshape(tedata_, (len(tedata),28))
    trdata_ = np.reshape(trdata_, (len(trdata),28))
    # telabel_ = np.reshape(telabel_, (len(tedata), 1))
    # trlabel_ = np.reshape(trlabel_, (len(trdata), 1))
    return trdata_, trlabel_, tedata_, telabel_


def create_svm(datamat, datalabel, decision='ovo'):
    classifier = svm.SVC(decision_function_shape=decision)
    classifier.fit(datamat, datalabel)
    return classifier

data = read_data()
train_data, train_label, test_data, test_label = get_data_and_labels(data, 0.2)
# print("training data is :\n", train_data)
# print("testing data is :\n", test_data)
print("training label is :\n", type(train_label))
# print("testing label is :\n", test_label)
st = time.clock()
clf = create_svm(train_data, train_label)
et = time.clock()
print("Training spent {:.4f}s.".format((et - st)))

pre_result = clf.predict(test_data)
score = clf.score(test_data, test_label)
print("Accuracy is: {:.6f}.".format(score))





