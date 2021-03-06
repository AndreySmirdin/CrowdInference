from typing import Iterable
from crowd_inference.model.annotation import Annotation
from crowd_inference.model.estimation import Estimation
from crowd_inference.truth_inference import NoFeaturesInference

import numpy as np
import sklearn.preprocessing


class DawidSkene(NoFeaturesInference):

    def __init__(self) -> None:
        super().__init__()
        self.predictions_ = {}

    def estimate(self) -> Iterable[Estimation]:
        return [Estimation(task, val) for task, val in self.predictions_.items()]

    def fit(self, annotations: Iterable[Annotation], max_iter=100):
        tasks = set(a.task for a in annotations)
        task_to_id = {task: i for i, task in enumerate(tasks)}

        workers = list(set(a.annotator for a in annotations))
        worker_to_id = {worker: i for i, worker in enumerate(workers)}
        
        values = list(set(a.value for a in annotations))
        value_to_id = {value: i for i, value in enumerate(values)}

        worker_annotations_values = [[] for _ in workers]
        worker_annotations_tasks = [[] for _ in workers]

        prediction_distr = np.zeros((len(tasks), len(values)))
        for a in annotations:
            a_id = worker_to_id[a.annotator]
            value_id = value_to_id[a.value]
            task_id = task_to_id[a.task]

            worker_annotations_values[a_id].append(value_id)
            worker_annotations_tasks[a_id].append(task_id)
            prediction_distr[task_id, value_id] += 1
        prediction_distr = sklearn.preprocessing.normalize(prediction_distr, axis=1, norm='l1')

        for i in range(len(worker_to_id)):
            worker_annotations_values[i] = np.array(worker_annotations_values[i])
            worker_annotations_tasks[i] = np.array(worker_annotations_tasks[i])


        prior = np.zeros(len(values))
        old_conf_mx = [np.zeros((len(values), len(values))) for _ in workers]

        for iter in range(max_iter):
            conf_mx = [np.zeros((len(values), len(values))) for _ in workers]
            for k in range(len(workers)):
                for j in range(len(values)):
                    np.add.at(conf_mx[k][:, j], worker_annotations_values[k], prediction_distr[worker_annotations_tasks[k], j])
                conf_mx[k] = np.transpose(conf_mx[k])
                conf_mx[k] = sklearn.preprocessing.normalize(conf_mx[k], axis=1, norm='l1')

            for j in range(len(values)):
                prior[j] = np.sum(prediction_distr[:, j]) / len(tasks)
            likelihood = np.ones((len(values), len(tasks)))

            for k in range(len(workers)):
                for j in range(len(values)):
                    np.multiply.at(likelihood[j, :], worker_annotations_tasks[k], conf_mx[k][j, worker_annotations_values[k]])
            likelihood = np.transpose(likelihood)

            logit = 1
            for i in range(len(tasks)):
                s = 0
                for j in range(len(values)):
                    prediction_distr[i, j] = prior[j] * likelihood[i, j]
                    s += prediction_distr[i, j]
                logit += np.log(s)
            print(f'Iter {iter:02}, logit: {logit / len(tasks):.6f}')

            prediction_distr = sklearn.preprocessing.normalize(prediction_distr, axis=1, norm='l1')

            converged = True
            for old, new in zip(old_conf_mx, conf_mx):
                if np.linalg.norm(old - new) > 0.0001:
                    converged = False

            if converged:
                break

            old_conf_mx = conf_mx
        self.predictions_ = {t: values[np.argmax(prediction_distr[i, :])] for t, i in task_to_id.items()}
