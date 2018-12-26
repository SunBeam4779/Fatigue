'''
Created on 2018年12月26日

@author: 37434
'''
from sklearn import svm

def create_svm(datamat, datalabel, decision='ovo'):
    classifier = svm.SVC(decision_function_shape=decision)
    classifier.fit(datamat, datalabel)
    return classifier