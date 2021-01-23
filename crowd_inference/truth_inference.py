from abc import abstractmethod
from typing import Iterable, Tuple, Set, Dict, List

import sklearn

from crowd_inference.model.annotation import Annotation
from crowd_inference.model.estimation import Estimation

import numpy as np


class TruthInference:
    tasks = []
    task_to_id = []
    workers = []
    worker_to_id = []
    values = []
    value_to_id = []

    logit_ = np.inf

    @abstractmethod
    def suffix(self):
        pass

    @abstractmethod
    def estimate(self) -> Iterable[Estimation]:
        pass

    def get_annotation_parameters(self, annotations: Iterable[Annotation]):
        self.tasks = np.array(sorted(list(set(a.task for a in annotations))))
        self.task_to_id = {task: i for i, task in enumerate(self.tasks)}
        self.workers = np.array(sorted(list(set(a.annotator for a in annotations))))
        self.worker_to_id = {worker: i for i, worker in enumerate(self.workers)}
        self.values = np.array(sorted(list(set(a.value for a in annotations))))
        self.value_to_id = {value: i for i, value in enumerate(self.values)}

    def get_majority_vote_probs(self, annotations: Iterable[Annotation]):
        prediction_distr = np.zeros((len(self.tasks), len(self.values)))
        for a in annotations:
            value_id = self.value_to_id[a.value]
            task_id = self.task_to_id[a.task]

            prediction_distr[task_id, value_id] += 1
        return sklearn.preprocessing.normalize(prediction_distr, axis=1, norm='l1')

    def get_worker_annotation(self, annotations: Iterable[Annotation]) -> Tuple[List[list], List[list]]:
        worker_annotations_values = [[] for _ in self.workers]
        worker_annotations_tasks = [[] for _ in self.workers]

        for a in annotations:
            worker_id = self.worker_to_id[a.annotator]
            task_id = self.task_to_id[a.task]
            value_id = self.value_to_id[a.value]
            worker_annotations_values[worker_id].append(value_id)
            worker_annotations_tasks[worker_id].append(task_id)

        for i in range(len(self.workers)):
            worker_annotations_values[i] = np.array(worker_annotations_values[i])
            worker_annotations_tasks[i] = np.array(worker_annotations_tasks[i])

        return worker_annotations_values, worker_annotations_tasks

    def get_loglike(self, mu, prior, likelihood):
        if prior.shape != likelihood.shape:
            prior = np.stack([prior] * len(likelihood))
        loglike = (mu * prior * np.exp(likelihood)).sum(axis=1)
        loglike = np.log(loglike).sum()
        return loglike / len(likelihood)

    def calculate_likelihoods(self, conf_mx, worker_annotations_values, worker_annotations_tasks):
        likelihood = np.zeros((len(self.values), len(self.tasks)))
        for k in range(len(self.workers)):
            for j in range(len(self.values)):
                val = conf_mx[k][j, worker_annotations_values[k]]
                np.add.at(likelihood[j, :], worker_annotations_tasks[k],
                          np.log(val, out=-np.ones_like(val) * np.inf, where=(val != 0)))
        likelihood = np.transpose(likelihood)
        return likelihood

    def calculate_conf_mx(self, mu, worker_annotations_values, worker_annotations_tasks):
        conf_mx = np.zeros((len(self.workers), len(self.values), len(self.values)))
        for k in range(len(self.workers)):
            for j in range(len(self.values)):
                np.add.at(conf_mx[k][:, j], worker_annotations_values[k],
                          mu[worker_annotations_tasks[k], j])
            conf_mx[k] = np.transpose(conf_mx[k])
            conf_mx[k] = sklearn.preprocessing.normalize(conf_mx[k], axis=1, norm='l1')

        return conf_mx


class NoFeaturesInference(TruthInference):
    @abstractmethod
    def fit(self, annotations: Iterable[Annotation]) -> np.ndarray:
        pass


class WithFeaturesInference(TruthInference):
    @abstractmethod
    def fit(self, annotations: Iterable[Annotation], features: Dict[str, np.ndarray]) -> np.ndarray:
        pass

    def evaluate_classifier(self, data):
        if data is not None:
            X, y = data
            y_pred = self.classifier.get_predictions(X, len(X))
            # It is supposed that labels are ordered alphabetically in y_pred
            self.losses.append(sklearn.metrics.log_loss(y, y_pred))
            if len(self.values) == 2:
                # self.accuracies.append(sklearn.metrics.roc_auc_score(y, y_pred[:, 1]))
                self.accuracies.append(sklearn.metrics.accuracy_score(y, self.values[np.argmax(y_pred, axis=1)]))

            else:
                self.accuracies.append(sklearn.metrics.accuracy_score(y, self.values[np.argmax(y_pred, axis=1)]))
        else:
            self.losses.append(0)
            self.accuracies.append(0)
