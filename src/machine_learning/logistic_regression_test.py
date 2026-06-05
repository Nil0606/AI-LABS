import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
from logistic_regression import LogisticRegression

bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=123)

def accuracy(y_true, y_predicted):
    accuracy = np.sum(y_true==y_predicted)/ len(y_true)
    return accuracy*100

regressor = LogisticRegression(lr=0.0001, n_iters=1000)
regressor.fit(X_train, y_train)
predictions = regressor.predict(X_test)
accuracy_test = accuracy(y_test, predictions)
print(f"LR classification accuracy: {accuracy_test}")
