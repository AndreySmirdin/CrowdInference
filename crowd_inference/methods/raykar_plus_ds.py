from typing import Iterable, Dict

import scipy

from crowd_inference.model.annotation import Annotation
from crowd_inference.model.estimation import Estimation
from crowd_inference.truth_inference import WithFeaturesInference
from crowd_inference.methods.raykar import update_w, get_predictions

import numpy as np
from scipy.special import expit
import sklearn.preprocessing


def sigmoid(x):
    return expit(x)


class RaykarPlusDs(WithFeaturesInference):

    def __init__(self) -> None:
        super().__init__()
        self.predictions_ = {}

    def __str__(self):
        return 'Raykar+DS'

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
        if n_values == 2:
            self.w = np.zeros(n_features)
        else:
            self.w = np.zeros((n_values, n_features))

        l = np.random.uniform(size=n_tasks)
        mu = self.get_majority_vote_probs(annotations)
        prior = np.zeros(n_values)
        worker_annotations_values, worker_annotations_tasks = self.get_worker_annotation(annotations)
        self.logit_ = []
        self.weights = []

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

            self.w = update_w(X, Xs, self.w, mu, n_tasks, n_values, n_features, lr)

            likelihood = np.ones((n_values, n_tasks))
            for k in range(n_workers):
                for j in range(n_values):
                    np.multiply.at(likelihood[j, :], worker_annotations_tasks[k],
                                   conf_mx[k][j, worker_annotations_values[k]])
            likelihood = np.transpose(likelihood)

            predictions = get_predictions(self.w, X, n_tasks, n_values)
            
#             r_weight = l.mean()
            for i in range(n_tasks):
                for j in range(n_values):
                    mu[i, j] = (predictions[i, j] * l[i] + prior[j] * (1 - l[i])) * likelihood[i, j]
            
            mu = sklearn.preprocessing.normalize(mu, axis=1, norm='l1')
            
            ds_sum = (mu * likelihood * prior).max(axis=1)
            r_sum = (mu * likelihood * predictions).max(axis=1)
            # Optimize lambda
            l = r_sum > ds_sum
            
            l_matrix = np.hstack([l.reshape(-1, 1)] * n_values)
            weights = predictions * l_matrix + np.stack([prior] * n_tasks) * (1 - l_matrix)
            loglike = self.get_loglike(mu, weights, likelihood)

#             assert not self.logit_ or loglike - self.logit_[-1] > -1e-4, self.logit_[-1] - loglike
            self.logit_.append(loglike)
            if iter % 10 == 0:
                print(f'Iter {iter:02}, logit: {loglike:.6f}')
                print(f'Average Raykar weight is {l.mean()}')

        self.predictions_ = {t: (self.values[np.argmax(mu[i, :])], mu[i], predictions[i], l[i], (ds_sum[i], r_sum[i]), likelihood[i]) for t, i in self.task_to_id.items()}
        print(f'Average Raykar weight is {l.mean()}')
        self.conf_mx = conf_mx
        
    def apply_classifier(self, features: np.ndarray) -> Dict[str, str]:
        res = []
        if len(self.w.shape) == 1:
            for prob in sigmoid(features @ self.w.T):
                res.append(self.values[0 if prob > 0.5 else 1])    
        else:
            for probs in np.dot(features, self.w.T):
                res.append(self.values[np.argmax(probs)])
        return np.array(res)
