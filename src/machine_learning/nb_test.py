import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt 

from nb import NavieBayes

def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy * 100

X, y = datasets.make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=123)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

nb = NavieBayes()
nb.fit(X_train, y_train)
predictions = nb.predict(X_test)

accuracy_test = accuracy(y_test, predictions)
print(f"Navie Bates Classification accuracy : {accuracy_test}%")