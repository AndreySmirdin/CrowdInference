from typing import Iterable, Dict

import scipy

from crowd_inference.methods.classifier import Classifier
from crowd_inference.model.annotation import Annotation
from crowd_inference.model.estimation import Estimation
from crowd_inference.truth_inference import WithFeaturesInference

import numpy as np
from scipy.special import expit
import sklearn.preprocessing


def sigmoid(x):
    return expit(x)


class RaykarPlusDs(WithFeaturesInference):

    def __init__(self, binary=False) -> None:
        super().__init__()
        self.predictions_ = {}
        self.binary = binary

    def __str__(self):
        return 'Raykar+DS'

    def estimate(self) -> Iterable[Estimation]:
        return [Estimation(task, val[0]) for task, val in self.predictions_.items()]

    def fit(self, annotations: Iterable[Annotation], features: Dict[str, np.ndarray], max_iter=200, lr=0.1,
            confidence_estimator=None):
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
        self.classifier = Classifier(n_features, n_values, lr)

        # l = np.random.uniform(size=n_tasks)
        l = np.zeros(n_tasks)
        mu = self.get_majority_vote_probs(annotations)
        prior = np.zeros(n_values)
        worker_annotations_values, worker_annotations_tasks = self.get_worker_annotation(annotations)
        self.logit_ = []
        self.weights = []
        self.mus = []
        self.cls = []
        self.conf_mxs = []
        self.priors = []

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

            self.classifier.update_w(X, Xs, mu, n_tasks)

            likelihood = np.ones((n_values, n_tasks))
            for k in range(n_workers):
                for j in range(n_values):
                    np.multiply.at(likelihood[j, :], worker_annotations_tasks[k],
                                   conf_mx[k][j, worker_annotations_values[k]])
            likelihood = np.transpose(likelihood)

            predictions = self.classifier.get_predictions(X, n_tasks)

            grads = np.zeros(n_tasks)
            mu_max = mu.argmax(axis=1)
            for i in range(n_tasks):
                grads[i] = np.linalg.norm((predictions[i, mu_max[i]] - mu[i, mu_max[i]]) * X[i])

            if not self.binary:
                if confidence_estimator is None:
                    l = np.exp(-grads)
                else:
                    for i in range(n_tasks):
                        confidence = confidence_estimator(grads[i])
                        l[i] = confidence

            else:
                for i in range(n_tasks):
                    l[i] = 1 if (likelihood[i, :] * predictions[i, :]).sum() > (
                                likelihood[i, :] * prior).sum() else 0
            l[:] = 0

            for i in range(n_tasks):
                for j in range(n_values):
                    mu[i, j] = (predictions[i, j] * l[i] + prior[j] * (1 - l[i])) * likelihood[i, j]

            mu = sklearn.preprocessing.normalize(mu, axis=1, norm='l1')

            l_matrix = np.hstack([l.reshape(-1, 1)] * n_values)
            weights = predictions * l_matrix + np.stack([prior] * n_tasks) * (1 - l_matrix)
            loglike = self.get_loglike(mu, weights, likelihood)

            #             assert not self.logit_ or loglike - self.logit_[-1] > -1e-4, self.logit_[-1] - loglike
            self.logit_.append(loglike)
            if iter % 10 == 0:
                print(f'Iter {iter:02}, logit: {loglike:.6f}')
                print(f'Average Raykar weight is {l.mean()}')
            self.mus.append(mu.copy())
            self.cls.append(predictions.copy())
            self.conf_mxs.append(conf_mx.copy())
            self.priors.append(prior.copy())
            self.weights.append(l.copy())

        self.mus = np.array(self.mus)
        self.cls = np.array(self.cls)
        self.conf_mxs = np.array(self.conf_mxs)
        self.priors = np.array(self.priors)
        self.weights = np.array(self.weights)

        self.predictions_ = {t: (
        self.values[np.argmax(mu[i, :])], mu[i], predictions[i], grads[i], likelihood[i], l[i], (l[i], 1 - l[i]), i) for
                             t, i in self.task_to_id.items()}
        print(f'Average Raykar weight is {l.mean()}')
        self.conf_mx = conf_mx

    def apply_classifier(self, features: np.ndarray) -> Dict[str, str]:
        return self.classifier.apply(features, self.values)
