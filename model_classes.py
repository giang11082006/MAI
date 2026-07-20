import numpy as np

class ZScoreNormalizer:
    def fit(self, X):
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_[self.std_ == 0] = 1
        return self

    def transform(self, X):
        return (X - self.mean_) / self.std_

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

class LinearRegression:
    def __init__(self, lr=0.01, epochs=10000):
        self.lr = lr
        self.epochs = epochs

    def fit(self, X, y):
        m, n = X.shape

        self.w = np.zeros(n)
        self.b = 0

        for _ in range(self.epochs):
            y_pred = X @ self.w + self.b
            error = y_pred - y

            dw = (1/m) * X.T @ error
            db = (1/m) * np.sum(error)

            self.w -= self.lr * dw
            self.b -= self.lr * db

    def predict(self, X):
        return X @ self.w + self.b