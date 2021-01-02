from typing import Iterable, Dict, List

from crowd_inference.methods.classifier import Classifier
from crowd_inference.model.annotation import Annotation
from crowd_inference.model.estimation import Estimation
from crowd_inference.truth_inference import WithFeaturesInference

import numpy as np
import sklearn.preprocessing

# def sigmoid(x):
#     return expit(x)
#
#
# def softmax(z):
#     return np.exp(z) / np.sum(np.exp(z), axis=1).reshape(-1, 1)
#
#
# def get_predictions(w, X, n_tasks, n_values):
#     if n_values > 2:
#         pre_vals = np.dot(X, w.T).reshape(-1, n_values)
#         return softmax(pre_vals)
#
#     predictions = np.zeros((n_tasks, n_values))
#     for i in range(n_tasks):
#         predictions[i, 0] = sigmoid(w.T @ X[i])
#         predictions[i, 1] = 1 - predictions[i, 0]
#     predictions = np.clip(predictions, 1e-6, 1 - 1e-6)
#     return predictions
#
#
# def update_w(X, Xs, w, mu, n_tasks, n_values, n_features, lr):
#     predictions = get_predictions(w, X, n_tasks, n_values)
#     g = np.zeros_like(w)
#     if n_values == 2:
#         for i in range(n_tasks):
#             g += (mu[i, 0] - predictions[i, 0]) * X[i]
#
#         H = np.zeros((n_features, n_features))
#         for i in range(n_tasks):
#             x = X[i].reshape(1, -1)
#             H -= predictions[i, 0] * predictions[i, 1] * x.T @ x
#         inv = np.linalg.pinv(H)
#         w -= lr * inv @ g
#     else:
#         invH = np.linalg.pinv(Xs * np.sum(predictions.T.dot(1 - predictions)))
#         grad = (predictions - mu).T.dot(X)
#         w -= grad.dot(invH)
#     return w


class Raykar(WithFeaturesInference):

    def __init__(self) -> None:
        super().__init__()
        self.predictions_ = {}

    def __str__(self):
        return 'Raykar'

    def estimate(self) -> Iterable[Estimation]:
        return [Estimation(task, val[0]) for task, val in self.predictions_.items()]

    def fit(self, annotations: Iterable[Annotation], features: Dict[str, np.ndarray], max_iter=200, lr=0.1):
        self.get_annotation_parameters(annotations)

        n_tasks = len(self.tasks)
        n_workers = len(self.workers)
        n_values = len(self.values)
        n_features = len(features[self.tasks[0]])
        print(f"Data has {n_features} features")

        X = np.zeros((n_tasks, n_features))
        for k, v in features.items():
            X[self.task_to_id[k]] = v
        Xs = X.T.dot(X)

        # if n_values == 2:
        #     self.w = np.zeros(n_features)
        # else:
        #     self.w = np.zeros((n_values, n_features))

        self.classifier = Classifier(n_features, n_values, lr)

        mu = self.get_majority_vote_probs(annotations)
        worker_annotations_values, worker_annotations_tasks = self.get_worker_annotation(annotations)
        self.logit_ = []
        self.mus = []
        self.cls = []

        for iter in range(max_iter):
            conf_mx = np.zeros((n_workers, n_values, n_values))
            for k in range(n_workers):
                for j in range(n_values):
                    np.add.at(conf_mx[k][:, j], worker_annotations_values[k],
                              mu[worker_annotations_tasks[k], j])
                conf_mx[k] = np.transpose(conf_mx[k])
                conf_mx[k] = sklearn.preprocessing.normalize(conf_mx[k], axis=1, norm='l1')

            # self.w = update_w(X, Xs, self.w, mu, n_tasks, n_values, n_features, lr)
            self.classifier.update_w(X, Xs, mu, n_tasks)

            likelihood = self.calculate_likelihoods(conf_mx, worker_annotations_values, worker_annotations_tasks)
            predictions = self.classifier.get_predictions(X, n_tasks)

            logit_ = 1
            for i in range(n_tasks):
                s = 0
                for j in range(n_values):
                    mu[i, j] = np.log(predictions[i, j]) + likelihood[i, j]
                    s += mu[i, j]
                # assert s > 0, predictions[i]
                # logit_ += np.log(s)
                mu[i] -= mu[i].max()
            logit_ /= n_tasks
            mu = np.exp(mu)
            mu = sklearn.preprocessing.normalize(mu, axis=1, norm='l1')

            loglike = self.get_loglike(mu, predictions, likelihood)
            #             assert not self.logit_ or loglike - self.logit_[-1] > -1e-4, self.logit_[-1] - loglike
            self.logit_.append(loglike)

            if iter % 10 == 0:
                # print(f'Iter {iter:02}, logit: {self.logit_:.6f}')
                print(f'Iter {iter:02}, logit: {loglike:.6f}')

            self.mus.append(mu.copy())
            self.cls.append(predictions.copy())

        grads = np.zeros(n_tasks)
        mu_max = mu.argmax(axis=1)
        for i in range(n_tasks):
            grads[i] = np.linalg.norm((predictions[i, mu_max[i]] - mu[i, mu_max[i]]) * X[i])
        self.mus = np.array(self.mus)
        self.cls = np.array(self.cls)
        self.predictions_ = {t: (self.values[np.argmax(mu[i, :])], mu[i], predictions[i], grads[i], likelihood[i]) for t, i in
                             self.task_to_id.items()}
        self.conf_mx = conf_mx

    def apply_classifier(self, features: np.ndarray) -> Dict[str, str]:
        print(features.shape)

        return self.classifier.apply(features, self.values)
