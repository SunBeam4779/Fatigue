import load_the_data
import tensorflow as tf
import time


DATA1 = "G:\\CAD\\pre_data1.csv"
DATA2 = "G:\\CAD\\pre_data4.csv"


training_data, test_data = load_the_data.load_data()
pre_data1 = load_the_data.load_test_data(DATA1)
pre_data2 = load_the_data.load_test_data(DATA2)

feature_columns = [tf.contrib.layers.real_valued_column("", dimension=28)]
dnn_clf = tf.contrib.learn.DNNClassifier(hidden_units=[60, 40], n_classes=2, feature_columns=feature_columns)
st = time.clock()
dnn_clf.fit(x=training_data.data, y=training_data.target, steps=1500)
et = time.clock()
score = dnn_clf.evaluate(x=test_data.data, y=test_data.target)["accuracy"]
print("training time is about:{0:4f}s".format(et-st))
print("Accuracy is:{0:f}".format(score))


def judge(state):
    if 0 in state:
        print("Its state is not fatigued")
    if 1 in state:
        print("Its state is fatigued")
    print("Prediction is:{0}".format(str(state)))


y1 = list(dnn_clf.predict(pre_data1.data))
y2 = list(dnn_clf.predict(pre_data2.data))
# print("Prediction 1 is:{0}".format(str(y1)))
# print("Prediction 2 is:{0}".format(str(y2)))

judge(y1)
judge(y2)
