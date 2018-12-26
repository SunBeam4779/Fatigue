import numpy as np
import os
import sys
from PIL import Image
import time
from sklearn import svm
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone

def get_file_list(path):
    return [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".png")]

def get_img_name_str(imgPath):
    return imgPath.split(os.path.sep)[-1]

##############################################################################
'''Image to Vector'''
def img2vector(imgFile):
    img = Image.open(imgFile).convert('L')
    img_arr = np.array(img, 'i')  # 28px * 28px 灰度图像
    img_normlization = np.round(img_arr/255)  # 对灰度值进行归一化
    img_arr2 = np.reshape(img_normlization, (1,-1))  # 1 * 400 矩阵
    return img_arr2

##############################################################################
'''turn every example into vector(Matrix)'''
def read_and_convert(imgFileList):
    dataLabel = []  # 存放类标签
    dataNum = len(imgFileList)
    dataMat = np.zeros((dataNum, 784))  # dataNum * 400 的矩阵
    for i in range(dataNum):
        imgNameStr = imgFileList[i]
        imgName = get_img_name_str(imgNameStr)  # 得到 数字_实例编号.png
        #print("imgName: {}".format(imgName))
        classTag = imgName.split(".")[0].split("_")[0]  # 得到 类标签(数字)
        #print("classTag: {}".format(classTag))
        dataLabel.append(classTag)
        dataMat[i,:] = img2vector(imgNameStr)
    return dataMat, dataLabel

#########################################################################################
'''读取训练数据'''
def read_all_data():
    cName = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
    train_data_path = "G:\\MNIST\\Image\\mnist_train\\0"
    flist = get_file_list(train_data_path)
    dataMat, dataLabel = read_and_convert(flist[:1000])
    for c in cName:
        train_data_path_ = "G:\\MNIST\\Image\\mnist_train\\" + c
        flist_ = get_file_list(train_data_path_)
        dataMat_, dataLabel_ = read_and_convert(flist_[:1000])
        dataMat = np.concatenate((dataMat, dataMat_), axis=0)
        dataLabel = np.concatenate((dataLabel, dataLabel_), axis=0)
    # print(dataMat.shape)
    # print(len(dataLabel))
    return dataMat, dataLabel


# create model
def create_svm(decision='ovr'):
    clf = svm.SVC(decision_function_shape=decision)
    #clf.fit(dataMat, dataLabel)
    return clf

def train_svm(dataMat,dataLabel,clf):
    clf.fit(dataMat,dataLabel)
    return clf

'''def cross_val(dataMat,dataLabel,clf):
    k=0
    skfolds=StratifiedKFold(n_splits=3)
    clone_clf=clone(clf)
    for train_index,test_index in skfolds.split(dataMat,dataLabel):
        X_train_folds=dataMat[train_index]
        Y_train_folds=dataLabel[train_index]
        X_test_folds=dataMat[test_index]
        Y_test_folds=dataLabel[test_index]
        if(k<3):
            clone_clf.fit(X_train_folds,Y_train_folds)
            Y_pred=clone_clf.predict(X_test_folds)
            n_correct=sum(Y_pred==Y_test_folds)
            print(n_correct/len(Y_pred))
        else:
            clone_clf.fit(dataMat,dataLabel)
        k+=1
    return clone_clf'''

trdataMat,trdataLabel=read_all_data()
print(trdataLabel.shape)
# clf = svm.SVC(decision_function_shape='ovr')
st = time.clock()
clf = create_svm(decision='ovr')
# cross_clf=cross_val(trdataMat,trdataLabel,clf)
clf.fit(trdataMat,trdataLabel)
et = time.clock()
print("Training spent {:.4f}s.".format((et - st)))


# 对10个数字进行分类测试
def main():
    tbasePath = "G:\\MNIST\\Image\\mnist_test\\"
    tcName = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    tst = time.clock()
    allErrCount = 0
    allErrorRate = 0.0
    allScore = 0.0
    for tcn in tcName:
        testPath = "G:\\MNIST\\Image\\mnist_test\\" + tcn
        # print("class " + tcn + " path is: {}.".format(testPath))
        tflist = get_file_list(testPath)
        # tflist
        tdataMat, tdataLabel = read_and_convert(tflist)
        print("test dataMat shape: {0}, test dataLabel len: {1} ".format(tdataMat.shape, len(tdataLabel)))

        # print("test dataLabel: {}".format(len(tdataLabel)))
        pre_st = time.clock()
        preResult = clf.predict(tdataMat)  # cross_clf.predict(tdataMat)
        pre_et = time.clock()
        print("Recognition  " + tcn + " spent {:.4f}s.".format((pre_et - pre_st)))
        # print("predict result: {}".format(len(preResult)))
        errCount = len([x for x in preResult if x != tcn])
        print("errorCount: {}.".format(errCount))
        allErrCount += errCount
        score_st = time.clock()
        score = clf.score(tdataMat, tdataLabel)  # cross_clf.score(tdataMat, tdataLabel)
        score_et = time.clock()
        print("computing score spent {:.6f}s.".format(score_et - score_st))
        allScore += score
        print("score: {:.6f}.".format(score))
        print("error rate is {:.6f}.".format((1 - score)))
        print("---------------------------------------------------------")

    tet = time.clock()
    print("Testing All class total spent {:.6f}s.".format(tet - tst))
    print("All error Count is: {}.".format(allErrCount))
    avgAccuracy = allScore / 10.0
    print("Average accuracy is: {:.6f}.".format(avgAccuracy))
    print("Average error rate is: {:.6f}.".format(1 - avgAccuracy))


main()
