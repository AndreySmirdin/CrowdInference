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

        self.losses = []
        self.accuracies = []

    def __str__(self):
        return 'DS'

    def suffix(self):
        return '_ds'

    def estimate(self) -> Iterable[Estimation]:
        return [Estimation(task, val[0]) for task, val in self.predictions_.items()]

    def fit(self, annotations: Iterable[Annotation], max_iter=200):
        self.get_annotation_parameters(annotations)
        tasks = sorted(list(set(a.task for a in annotations)))
        task_to_id = {task: i for i, task in enumerate(tasks)}

        workers = sorted(list(set(a.annotator for a in annotations)))
        worker_to_id = {worker: i for i, worker in enumerate(workers)}

        values = sorted(list(set(a.value for a in annotations)))
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
        self.logit_ = []
        self.mus = []
        self.priors = []

        for iter in range(max_iter):
            conf_mx = self.calculate_conf_mx(prediction_distr, worker_annotations_values, worker_annotations_tasks)
            # conf_mx = [np.zeros((len(values), len(values))) for _ in workers]
            # for k in range(len(workers)):
            #     for j in range(len(values)):
            #         np.add.at(conf_mx[k][:, j], worker_annotations_values[k],
            #                   prediction_distr[worker_annotations_tasks[k], j])
            #         conf_mx[k][:, j] += conf_mx[k][:, j].sum() * 0.005
            #     conf_mx[k] = np.transpose(conf_mx[k])
            #     conf_mx[k] = sklearn.preprocessing.normalize(conf_mx[k], axis=1, norm='l1')

            for j in range(len(values)):
                prior[j] = np.sum(prediction_distr[:, j]) / len(tasks)

            likelihood = self.calculate_likelihoods(conf_mx, worker_annotations_values, worker_annotations_tasks)

            for i in range(len(tasks)):
                s = 0
                for j in range(len(values)):
                    prediction_distr[i, j] = np.log(prior[j]) + likelihood[i, j]
                    s += prediction_distr[i, j]

                prediction_distr[i] -= prediction_distr[i].max()
            prediction_distr = np.exp(prediction_distr)
            prediction_distr = sklearn.preprocessing.normalize(prediction_distr, axis=1, norm='l1')

            loglike = self.get_loglike(prediction_distr, prior, likelihood)
            # assert not self.logit_ or self.logit_[-1] < loglike
            self.logit_.append(loglike)

            if iter % (max_iter // 5) == 0:
                print(f'Iter {iter:02}, logit: {loglike:.6f}')

            converged = True
            for old, new in zip(old_conf_mx, conf_mx):
                if np.linalg.norm(old - new) > 0.0001:
                    converged = False

            if converged:
                break

            old_conf_mx = conf_mx
            self.mus.append(prediction_distr.copy())
            self.priors.append(prior.copy())


        print(self.priors[-1])
        # print('---------------')
        # for a in annotations:
        #     if a.task == 't109':
        #         print(a.annotator, conf_mx[self.worker_to_id[a.annotator]][1])
        #         print(a.annotator, conf_mx[self.worker_to_id[a.annotator]][2])
        self.mus = np.array(self.mus)
        self.priors = np.array(self.priors)

        self.predictions_ = {t: (values[np.argmax(prediction_distr[i, :])], prediction_distr[i, :], None, None, likelihood[i]) for t, i in
                             task_to_id.items()}
        self.conf_mx = np.array(conf_mx)
