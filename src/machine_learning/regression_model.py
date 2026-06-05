import numpy as np


class BaseRegression:

    def __init__(self, lr=0.001, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for _ in range(self.n_iters):

            y_predicted = self._approximation(
                X,
                self.weights,
                self.bias
            )

            dw = (1 / n_samples) * np.dot(
                X.T,
                (y_predicted - y)
            )

            db = (1 / n_samples) * np.sum(
                y_predicted - y
            )

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        return self._predict(
            X,
            self.weights,
            self.bias
        )

    def _approximation(self, X, w, b):
        raise NotImplementedError()

    def _predict(self, X, w, b):
        raise NotImplementedError()


class LinearRegression(BaseRegression):

    def _approximation(self, X, w, b):
        return np.dot(X, w) + b

    def _predict(self, X, w, b):
        return np.dot(X, w) + b


class LogisticRegression(BaseRegression):

    def _approximation(self, X, w, b):
        linear_model = np.dot(X, w) + b
        return self._sigmoid(linear_model)

    def _predict(self, X, w, b):
        linear_model = np.dot(X, w) + b
        probabilities = self._sigmoid(linear_model)

        return np.array(
            [1 if p > 0.5 else 0 for p in probabilities]
        )

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))


if __name__ == '__main__':

    X = np.array([
        [1, 2],
        [2, 3],
        [3, 4],
        [4, 5]
    ])

    y = np.array([0, 0, 1, 1])

    model = LogisticRegression(lr=0.01, n_iters=1000)

    model.fit(X, y)

    print('Weights:', model.weights)
    print('Bias:', model.bias)
    print('Predictions:', model.predict(X))
