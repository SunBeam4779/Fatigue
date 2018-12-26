'''
Created on 2018年12月26日

@author: 37434
'''
import numpy as np
import pandas as pd
from builtins import len


PATH = "G:\\CAD\\data_1.csv"


def read_data(path=PATH):
    data = pd.read_csv(path, encoding='ANSI', delimiter=",")
    data = data.drop([0])
    data_mat = np.array(data)
    data_mat = np.delete(data_mat, 0, axis=1)
    return data_mat


def get_data_and_labels(data, test_ratio):
    datanum = int(len(data)/2)
    datalabel_1 = []
    datalabel_0 = []
    test_set_size = int(datanum*test_ratio)
    data_label = []
    data_ = data[:, :]
    m, n = np.shape(data)
    # label_ = data[:, -1]
#     data_0 = data[:datanum+1, :28]
#     label_0 = data[:datanum+1, -1]
#     data_1 = data[datanum+1:, :28]
#     label_1 = data[datanum+1:, -1]
#     for i in range(len(data_1)):
#         datalabel_1.append(label_1[i])
#         for j in range(28):
#             data_1[i][j] = float(data_1[i][j])
#     for i in range(len(data_0)):
#         datalabel_0.append(label_0[i])
#         for j in range(28):
#             data_0[i][j] = float(data_0[i][j])
#     for i in range(len(data_)):
#         data_label.append(label_[i])
#         for j in range(28):
#             data_[i][j] = float(data_[i][j])
#     trdata_0 = data_0[test_set_size:]
#     trdata_1 = data_1[test_set_size:]
#     tedata_0 = data_0[:test_set_size]
#     tedata_1 = data_1[:test_set_size]
#     trlabel_0 = label_0[test_set_size:]
#     trlabel_1 = label_1[test_set_size:]
#     telabel_0 = label_0[:test_set_size]
#     telabel_1 = label_1[:test_set_size]
#     trdata = np.concatenate((trdata_0, trdata_1), axis=0)
#     tedata = np.concatenate((tedata_0, tedata_1), axis=0)
#     trlabel = np.concatenate((trlabel_0, trlabel_1), axis=0)
#     telabel = np.concatenate((telabel_0, telabel_1), axis=0)
    for i in range(m):
        for j in range(n):
            data[i][j] = float(data[i][j])
    # print(data_label)
    trdata = np.array(data[:80])
    trlabel = np.array(data_label[:80])
    tedata = np.array(data[79:])
    telabel = np.array(data_label[79:])
    return trdata, trlabel, tedata, telabel
    return trdata, tedata
    