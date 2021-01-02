import numpy as np
from scipy.special import expit


def sigmoid(x):
    return expit(x)


def softmax(z):
    return np.exp(z) / np.sum(np.exp(z), axis=1).reshape(-1, 1)


class Classifier:
    def __init__(self, n_features, n_classes, lr):
        self.n_features = n_features
        self.n_classes = n_classes
        self.lr = lr

        if n_classes == 2:
            self.w = np.zeros(n_features)
        else:
            self.w = np.zeros((n_classes, n_features))

    def update_w(self, X, Xs, mu, n_tasks):
        predictions = self.get_predictions(X, n_tasks)
        g = np.zeros_like(self.w)
        if self.n_classes == 2:
            for i in range(n_tasks):
                g += (mu[i, 0] - predictions[i, 0]) * X[i]

            H = np.zeros((self.n_features, self.n_features))
            for i in range(n_tasks):
                x = X[i].reshape(1, -1)
                H -= predictions[i, 0] * predictions[i, 1] * x.T @ x
            inv = np.linalg.pinv(H)
            self.w -= self.lr * inv @ g
        else:
            invH = np.linalg.pinv(Xs * np.sum(predictions.T.dot(1 - predictions)))
            grad = (predictions - mu).T.dot(X)
            # print(predictions[-3:])
            self.w -= grad.dot(invH) * self.lr
            # print(np.abs(grad.dot(invH)).sum())
            # print(self.w)

    def get_predictions(self, X, n_tasks):
        if self.n_classes > 2:
            pre_vals = np.dot(X, self.w.T).reshape(-1, self.n_classes)
            return softmax(pre_vals)

        predictions = np.zeros((n_tasks, self.n_classes))
        for i in range(n_tasks):
            predictions[i, 0] = sigmoid(self.w.T @ X[i])
            predictions[i, 1] = 1 - predictions[i, 0]
        predictions = np.clip(predictions, 1e-6, 1 - 1e-6)
        return predictions

    def get_scores(self, features):
        if len(self.w.shape) == 1:
            return sigmoid(features @ self.w.T)
        return np.dot(features, self.w.T)

    def apply(self, features: np.ndarray, values):
        res = []
        if len(self.w.shape) == 1:
            for prob in sigmoid(features @ self.w.T).ravel():
                res.append(values[0 if prob > 0.5 else 1])
        else:
            for probs in np.dot(features, self.w.T):
                res.append(values[np.argmax(probs)])
        return np.array(res)
