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
    def estimate(self) -> Iterable[Estimation]:
        pass

    def get_annotation_parameters(self, annotations: Iterable[Annotation]):
        self.tasks = sorted(list(set(a.task for a in annotations)))
        self.task_to_id = {task: i for i, task in enumerate(self.tasks)}
        self.workers = sorted(list(set(a.annotator for a in annotations)))
        self.worker_to_id = {worker: i for i, worker in enumerate(self.workers)}
        self.values = sorted(list(set(a.value for a in annotations)))
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
        loglike = (mu * prior * likelihood).sum(axis=1)
        loglike = np.log(loglike).sum()
        return loglike / len(likelihood)


class NoFeaturesInference(TruthInference):
    @abstractmethod
    def fit(self, annotations: Iterable[Annotation]) -> np.ndarray:
        pass


class WithFeaturesInference(TruthInference):
    @abstractmethod
    def fit(self, annotations: Iterable[Annotation], features: Dict[str, np.ndarray]) -> np.ndarray:
        pass
