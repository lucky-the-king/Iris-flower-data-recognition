# This app will be able to recognize Iris flowers
# Collect data ( Supervised Learning: data is labaled )
# Pick the model
# Train the model
# Test the model

import tensorflow.contrib.learn as skflow
from sklearn import datasets, metrics
from sklearn import model_selection

# get the data
iris = datasets.load_iris()

# split the data into test and train
x_train, x_test, y_train, y_test = model_selection.train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42)

# Feature columns
feature_columns = skflow.infer_real_valued_columns_from_input(x_train)

# create model
# Linear Classifier
classifier = skflow.LinearClassifier(feature_columns=feature_columns, n_classes=3, model_dir='/tmp/tf/linear/')
# fit the model
classifier.fit(iris.data, iris.target)
# get the accuracy
score = metrics.accuracy_score(y_test, classifier.predict(x_test))
# print the test accuracy
print("Accuracy: %f" % score)
