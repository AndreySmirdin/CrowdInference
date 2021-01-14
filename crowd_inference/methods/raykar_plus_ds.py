from typing import Iterable, Dict

import scipy

from crowd_inference.methods.classifier import Classifier
from crowd_inference.model.annotation import Annotation
from crowd_inference.model.estimation import Estimation
from crowd_inference.truth_inference import WithFeaturesInference

import numpy as np
import sklearn.preprocessing


class RaykarPlusDs(WithFeaturesInference):

    def __init__(self, binary=False) -> None:
        super().__init__()
        self.predictions_ = {}
        self.binary = binary

    def __str__(self):
        return 'Raykar+DS'

    def suffix(self):
        return '_rds' + ('_binary' if self.binary else '')

    def estimate(self) -> Iterable[Estimation]:
        return [Estimation(task, val[0]) for task, val in self.predictions_.items()]

    def fit(self, annotations: Iterable[Annotation], features: Dict[str, np.ndarray], max_iter=200, lr=0.1,
            confidence_estimator=None, n_cls=7):
        self.get_annotation_parameters(annotations)

        n_tasks = len(self.tasks)
        n_values = len(self.values)
        n_features = len(features[self.tasks[0]])
        print(f"Data has {n_features} features")

        X = np.zeros((n_tasks, n_features))
        for k, v in features.items():
            X[self.task_to_id[k]] = v

        Xs = X.T.dot(X)

        # datasets = np.random.randint(0, n_tasks, (n_cls, n_tasks // 2))
        # X_boosted = np.zeros((n_cls, n_tasks // 2, n_features))
        # Xs_boosted = []
        # for i in range(n_cls):
        #     X_boosted[i] = X[datasets[i]]
        #     Xs_boosted.append(X_boosted[i].T.dot(X_boosted[i]))
        
        self.classifier = Classifier(n_features, n_values, lr)
        # self.classifiers = [Classifier(n_features, n_values, lr) for _ in range(n_cls)]

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
            conf_mx = self.calculate_conf_mx(mu, worker_annotations_values, worker_annotations_tasks)

            for j in range(n_values):
                prior[j] = np.sum(mu[:, j]) / n_tasks

            self.classifier.update_w(X, Xs, mu)
            # for i, classifier in enumerate(self.classifiers):
            #     classifier.update_w(X_boosted[i], Xs_boosted[i], mu[datasets[i]])

            likelihood = self.calculate_likelihoods(conf_mx, worker_annotations_values, worker_annotations_tasks)
            predictions = self.classifier.get_predictions(X, n_tasks)

            # predictions_other = np.zeros((n_cls, n_tasks, n_values))
            # for i in range(n_cls):
            #     predictions_other[i] = self.classifiers[i].get_predictions(X, n_tasks)

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
                        if iter < max_iter * 0.85:
                            l[:] = 1

            else:
                for i in range(n_tasks):
                    l[i] = 1 if (likelihood[i, :] * predictions[i, :]).sum() > (
                                likelihood[i, :] * prior).sum() else 0
            # l[:] = 1 - (predictions_other[:, :, 0].max(axis=0) - predictions_other[:, :, 0].min(axis=0)) * 2

            for i in range(n_tasks):
                for j in range(n_values):
                    mu[i, j] = np.log(predictions[i, j] * l[i] + prior[j] * (1 - l[i])) + likelihood[i, j]

            mu = np.exp(mu)
            mu = sklearn.preprocessing.normalize(mu, axis=1, norm='l1')

            l_matrix = np.hstack([l.reshape(-1, 1)] * n_values)
            weights = predictions * l_matrix + np.stack([prior] * n_tasks) * (1 - l_matrix)
            loglike = self.get_loglike(mu, weights, likelihood)

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
