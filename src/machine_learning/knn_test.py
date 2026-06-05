import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 
from matplotlib.colors import ListedColormap
cmap = ListedColormap(['#FF0000','#00ff00','#0000ff'])


iris = datasets.load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1234)

print(f"x = {X[0,0]}, y = {X[0,1]} color = {y[0]}")
# print(X_train[0])
# print(X_test[0])

# print(y_train.shape)
# print(y_test.shape)

# plt.figure()
# plt.scatter(X[0:1, 0], X[0:1, 1], c=y[0:1], cmap=cmap, s=20)
# plt.show()


from knn import KNN

clf = KNN(k=3)
clf.fit(X_train, y_train)

predictions = clf.predict(X_test)

accuracy = np.sum(predictions == y_test) / len(y_test)
accuracy *= 100
print(F"{accuracy=}")
