from typing import Iterable, Dict

from crowd_inference.methods.classifier import Classifier
from crowd_inference.model.annotation import Annotation
from crowd_inference.model.estimation import Estimation
from crowd_inference.truth_inference import WithFeaturesInference

import numpy as np
import sklearn.preprocessing


class Raykar(WithFeaturesInference):

    def __init__(self) -> None:
        super().__init__()
        self.predictions_ = {}

        self.losses = []
        self.accuracies = []

    def __str__(self):
        return 'Raykar'

    def suffix(self):
        return '_r'

    def estimate(self) -> Iterable[Estimation]:
        return [Estimation(task, val[0]) for task, val in self.predictions_.items()]

    def fit(self, annotations: Iterable[Annotation], features: Dict[str, np.ndarray], max_iter=200, lr=0.1, test=None):
        self.get_annotation_parameters(annotations)

        n_tasks = len(self.tasks)
        n_values = len(self.values)
        n_features = len(features[self.tasks[0]])
        print(f"Data has {n_features} features")

        X = np.zeros((n_tasks, n_features))
        for k, v in features.items():
            X[self.task_to_id[k]] = v

        self.classifier = Classifier(n_features, n_values, lr)

        mu = self.get_majority_vote_probs(annotations)
        worker_annotations_values, worker_annotations_tasks = self.get_worker_annotation(annotations)
        self.logit_ = []
        self.mus = []
        self.cls = []
        self.grads_history = []

        for iter in range(max_iter):
            conf_mx = self.calculate_conf_mx(mu, worker_annotations_values, worker_annotations_tasks)

            self.classifier.update_w(X, mu)

            likelihood = self.calculate_likelihoods(conf_mx, worker_annotations_values, worker_annotations_tasks)
            predictions = self.classifier.get_predictions(X, n_tasks)

            logit_ = 1
            for i in range(n_tasks):
                s = 0
                for j in range(n_values):
                    mu[i, j] = np.log(predictions[i, j]) + likelihood[i, j]
                    s += mu[i, j]
                mu[i] -= mu[i].max()

            logit_ /= n_tasks
            mu = np.exp(mu)
            mu = sklearn.preprocessing.normalize(mu, axis=1, norm='l1')

            loglike = self.get_loglike(mu, predictions, likelihood)
            self.logit_.append(loglike)

            if iter % (max_iter // 5) == 0:
                print(f'Iter {iter:02}, logit: {loglike:.6f}')

            # print(predictions.max())
            grads = np.linalg.norm((1 - predictions.max(axis=1))[:, None] * X, axis=1) ** 2
            # mu_max = mu.argmax(axis=1)
            # for i in range(n_tasks):
            #     grads[i] = np.linalg.norm((predictions[i]) * X[i]) ** 2
            self.grads_history.append(grads)

            self.mus.append(mu.copy())
            self.cls.append(predictions.copy())

            self.evaluate_classifier(test)

        self.grads_history = np.array(self.grads_history)
        self.mus = np.array(self.mus)
        self.cls = np.array(self.cls)
        self.predictions_ = {t: (self.values[np.argmax(mu[i, :])], mu[i], predictions[i], self.grads_history[:, i], likelihood[i]) for t, i in
                             self.task_to_id.items()}
        self.conf_mx = conf_mx

    def apply_classifier(self, features: np.ndarray) -> Dict[str, str]:
        print(features.shape)
        return self.classifier.apply(features, self.values)
