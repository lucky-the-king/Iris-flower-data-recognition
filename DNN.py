# This app will be able to recognize Iris flowers
# Collect data ( Supervised Learning: data is labaled )
# Pick the model
# Train the model
# Test the model



import tensorflow.contrib.learn as learn
from sklearn import datasets, metrics

# load data
iris = datasets.load_iris()

# split the data into test and train
x_train, x_test, y_train, y_test = cross_validation.train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42)

# Feature columns is required for new versions
feature_columns = skflow.infer_real_valued_columns_from_input(x_train)s

# create model
# Deep Neural Network
classifier = skflow.DNNClassifier(feature_columns=feature_columns, hidden_units=[10, 20, 10], n_classes=3,model_dir='/tmp/tf/mlp/')
# fit the model
classifier.fit(iris.data, iris.target)
# get the accuracy
score = metrics.accuracy_score(iris.target, classifier.predict(iris.data))
# print the test accuracy
print("Accuracy: %f" % score)
