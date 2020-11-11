from typing import Iterable, Dict

import scipy

from crowd_inference.model.annotation import Annotation
from crowd_inference.model.estimation import Estimation
from crowd_inference.truth_inference import WithFeaturesInference

import numpy as np
import sklearn.preprocessing


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class RaykarPlusDs(WithFeaturesInference):

    def __init__(self) -> None:
        super().__init__()
        self.predictions_ = {}

    def __str__(self):
        return 'Raykar+DS'

    def estimate(self) -> Iterable[Estimation]:
        return [Estimation(task, val) for task, val in self.predictions_.items()]

    def fit(self, annotations: Iterable[Annotation], features: Dict[str, np.ndarray], max_iter=200, lr=0.1):
        self.get_annotation_parameters(annotations)

        n_tasks = len(self.tasks)
        n_workers = len(self.workers)
        n_values = len(self.values)
        n_features = len(features[self.tasks[0]])
        print(f"Data has {n_features} features")
        assert n_values == 2, "Well, it's too complicated."

        X = np.zeros((n_tasks, n_features))
        for k, v in features.items():
            X[self.task_to_id[k]] = v

        self.w = np.random.randn(n_features)

        l = np.random.uniform(size=n_tasks)
        mu = self.get_majority_vote_probs(annotations)
        prior = np.zeros(n_values)
        worker_annotations_values, worker_annotations_tasks = self.get_worker_annotation(annotations)

        for iter in range(max_iter):
            conf_mx = np.zeros((n_workers, n_values, n_values))
            for k in range(n_workers):
                for j in range(n_values):
                    np.add.at(conf_mx[k][:, j], worker_annotations_values[k],
                              mu[worker_annotations_tasks[k], j])
                conf_mx[k] = np.transpose(conf_mx[k])
                conf_mx[k] = sklearn.preprocessing.normalize(conf_mx[k], axis=1, norm='l1')

            for j in range(n_values):
                prior[j] = np.sum(mu[:, j]) / n_tasks

            g = np.zeros_like(self.w)
            for i in range(n_tasks):
                g += (mu[i, 0] - sigmoid(self.w.T @ X[i])) * X[i]

            H = np.zeros((n_features, n_features))
            for i in range(n_tasks):
                s = sigmoid(self.w.T @ X[i])
                x = X[i].reshape(1, -1)
                H -= s * (1 - s) * x.T @ x
            inv = scipy.linalg.inv(H)
            self.w = self.w - lr * inv @ g

            likelihood = np.ones((n_values, n_tasks))
            for k in range(n_workers):
                for j in range(n_values):
                    np.multiply.at(likelihood[j, :], worker_annotations_tasks[k],
                                   conf_mx[k][j, worker_annotations_values[k]])
            likelihood = np.transpose(likelihood)

            predictions = np.zeros((n_tasks, n_values))
            for i in range(n_tasks):
                predictions[i, 0] = sigmoid(self.w.T @ X[i])
                predictions[i, 1] = 1 - predictions[i, 0]

            self.logit_ = 1
            for i in range(n_tasks):
                s = 0
                r_sum = 0
                r_ds_sum = 0
                for j in range(n_values):
                    mu[i, j] = (predictions[i, j] * l[i] + prior[j] * (1 - l[i])) * likelihood[i, j]
                    s += mu[i, j]
                    r_sum += likelihood[i, j] * predictions[i, j]
                    r_ds_sum += likelihood[i, j] * (predictions[i, j] + prior[j])
                self.logit_ += np.log(s)
                l[i] = r_sum / r_ds_sum
            self.logit_ /= n_tasks
            print(f'Iter {iter:02}, logit: {self.logit_:.6f}')

            mu = sklearn.preprocessing.normalize(mu, axis=1, norm='l1')

        self.predictions_ = {t: self.values[np.argmax(mu[i, :])] for t, i in self.task_to_id.items()}
        print(f'Average Raykar weight is {l.mean()}')

    def apply_classifier(self, features: Dict[str, np.ndarray]) -> Dict[str, str]:
        result = {}
        for k, x in features.items():
            result[k] = self.values[0 if sigmoid(self.w.T @ np.array(x, dtype='float')) >= 0.5 else 1]

        return result
