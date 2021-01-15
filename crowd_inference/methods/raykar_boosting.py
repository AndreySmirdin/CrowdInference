from typing import Iterable, Dict

from crowd_inference.methods.classifier import Classifier
from crowd_inference.model.annotation import Annotation
from crowd_inference.model.estimation import Estimation
from crowd_inference.truth_inference import WithFeaturesInference

import numpy as np
import sklearn.preprocessing


class RaykarWithBoosting(WithFeaturesInference):

    def __init__(self) -> None:
        super().__init__()
        self.predictions_ = {}

    def __str__(self):
        return 'RaykarBoosting'

    def suffix(self):
        return '_rb'

    def estimate(self) -> Iterable[Estimation]:
        return [Estimation(task, val[0]) for task, val in self.predictions_.items()]

    def fit(self, annotations: Iterable[Annotation], features: Dict[str, np.ndarray], max_iter=200, lr=0.1, n_cls=5):
        self.get_annotation_parameters(annotations)

        n_tasks = len(self.tasks)
        n_values = len(self.values)
        n_features = len(features[self.tasks[0]])
        datasets = np.random.randint(0, n_tasks, (n_cls, n_tasks // 2))
        # for i in range(3):
        # datasets[0] = np.arange(n_tasks)
        print(f"Data has {n_features} features")

        X = np.zeros((n_tasks, n_features))
        for k, v in features.items():
            X[self.task_to_id[k]] = v
        X_boosted = np.zeros((n_cls, n_tasks // 2, n_features))
        Xs_boosted = []
        for i in range(n_cls):
            X_boosted[i] = X[datasets[i]]
            Xs_boosted.append(X_boosted[i].T.dot(X_boosted[i]))

        self.classifiers = [Classifier(n_features, n_values, lr) for _ in range(n_cls)]

        mu = self.get_majority_vote_probs(annotations)
        worker_annotations_values, worker_annotations_tasks = self.get_worker_annotation(annotations)
        self.logit_ = []
        self.mus = []
        self.cls = []

        for iter in range(max_iter):
            conf_mx = self.calculate_conf_mx(mu, worker_annotations_values, worker_annotations_tasks)

            for i, classifier in enumerate(self.classifiers):
                classifier.update_w(X_boosted[i], Xs_boosted[i], mu[datasets[i]])

            likelihood = self.calculate_likelihoods(conf_mx, worker_annotations_values, worker_annotations_tasks)
            predictions = np.zeros((n_cls, n_tasks, n_values))
            for i in range(n_cls):
                predictions[i] = self.classifiers[i].get_predictions(X, n_tasks)
            # predictions = predictions.prod(axis=0)
            # predictions = sklearn.preprocessing.normalize(predictions, axis=1, norm='l1')
            predictions_agg = predictions.mean(axis=0)
            logit_ = 1
            for i in range(n_tasks):
                for j in range(n_values):
                    mu[i, j] = np.log(predictions_agg[i, j]) + likelihood[i, j]
                mu[i] -= mu[i].max()

            logit_ /= n_tasks
            mu = np.exp(mu)
            mu = sklearn.preprocessing.normalize(mu, axis=1, norm='l1')

            loglike = self.get_loglike(mu, predictions_agg, likelihood)
            self.logit_.append(loglike)

            if iter % (max_iter // 5) == 0:
                print(f'Iter {iter:02}, logit: {loglike:.6f}')

            self.mus.append(mu.copy())
            self.cls.append(predictions.copy())

        grads = np.zeros(n_tasks)
        mu_max = mu.argmax(axis=1)
        # for i in range(n_tasks):
        #     grads[i] = np.linalg.norm((predictions[i, mu_max[i]] - mu[i, mu_max[i]]) * X[i])
        self.mus = np.array(self.mus)
        self.cls = np.array(self.cls)
        self.predictions_ = {t: (self.values[np.argmax(mu[i, :])], mu[i], predictions[:, i], grads[i], likelihood[i]) for t, i in
                             self.task_to_id.items()}
        self.conf_mx = conf_mx

    def apply_classifier(self, features: np.ndarray) -> Dict[str, str]:
        print(features.shape)
        return self.classifier.apply(features, self.values)
