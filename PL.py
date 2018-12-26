# import test
import test.load_the_data
# import test.KNN
import tensorflow as tf
# from tensorflow._api.v1 import feature_column
# from Tools.demo.sortvisu import steps
from sklearn.metrics.scorer import accuracy_scorer



# data = test.load_the_data.read_data()
# training_data, training_label, test_data, test_label = \
#     test.load_the_data.get_data_and_labels(data, 0.2)
training_data, test_data = test.load_the_data.load_data()

feature_columns = [tf.contrib.layers.real_valued_column("", dimension=28)]
dnn_clf = tf.contrib.learn.DNNClassifier(hidden_units=[60, 50], n_classes=2, feature_columns=feature_columns)
dnn_clf.fit(x=training_data.data, y=training_data.target, steps=100)
score = dnn_clf.evaluate(x=test_data.data, y=test_data.target)["accuracy"]
print("Accuracy is:{:.4f}".format(score))

