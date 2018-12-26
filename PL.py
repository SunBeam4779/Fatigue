import test
import test.load_the_data
import test.KNN
import tensorflow as tf
from tensorflow._api.v1 import feature_column
from Tools.demo.sortvisu import steps
from sklearn.metrics import accuracy_score


data = test.load_the_data.read_data()
training_data, training_label, test_data, test_label = \
    test.load_the_data.get_data_and_labels(data, 0.2)

feature_columns = tf.contrib.learn.infer_real_valued_columns_from_input(training_data)
dnn_clf = tf.contrib.learn.DNNClassifier(hidden_units=[60, 50], n_classes=1, feature_columns=feature_columns)
dnn_clf.fit(x=training_data, y=training_label, batch_size=1, steps=100)
pred_label = list(dnn_clf.predict(test_data))
print(accuracy_score(test_label,pred_label))

