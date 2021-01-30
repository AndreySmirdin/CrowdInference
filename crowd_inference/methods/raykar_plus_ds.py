import bisect
from typing import Iterable, Dict, List

from scipy.stats import binom
from statsmodels.stats.proportion import proportion_confint

from benchmark import get_rnd_cls_accuracy
from crowd_inference.methods.classifier import Classifier
from crowd_inference.model.annotation import Annotation
from crowd_inference.model.estimation import Estimation
from crowd_inference.truth_inference import WithFeaturesInference

import numpy as np
import sklearn.preprocessing

from tests.data_provider import DataProvider


class GradientBucket:
    def __init__(self, grads, correctness):
        self.center = (grads[0] + grads[-1]) * 0.5
        self.width = grads[-1] - grads[0]
        self.right_end = self.center + self.width * 0.5
        self.correct = correctness.sum()
        self.size = len(grads)
        self.accuracy = self.correct / self.size
        low, up = proportion_confint(self.correct, self.size)
        self.conf_interval = (up - low) * 0.5


class RaykarPlusDs(WithFeaturesInference):

    def __init__(self, binary=False) -> None:
        super().__init__()
        self.predictions_ = {}
        self.binary = binary

        self.losses = []
        self.accuracies = []

    def __str__(self):
        return 'Raykar+DS' + ('_binary' if self.binary else '')

    def suffix(self):
        return '_rds' + ('_binary' if self.binary else '')

    def estimate(self) -> Iterable[Estimation]:
        return [Estimation(task, val[0]) for task, val in self.predictions_.items()]

    def fit(self, annotations: Iterable[Annotation], features: Dict[str, np.ndarray], data: DataProvider = None,
            max_iter=200, lr=0.1, confidence_estimator=None, n_bucket=None, test=None, axes=None):
        self.get_annotation_parameters(annotations)

        n_tasks = len(self.tasks)
        n_values = len(self.values)
        n_features = len(features[self.tasks[0]])
        print(f"Data has {n_features} features")
        if n_bucket is None:
            n_bucket = n_tasks // 10

        X = np.zeros((n_tasks, n_features))
        for k, v in features.items():
            X[self.task_to_id[k]] = v

        self.classifier = Classifier(n_features, n_values, lr)

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

        ground_truth_ids = self.get_ground_truth_ids(data)
        rnd_accuracy = get_rnd_cls_accuracy(ground_truth_ids)
        # rnd_accuracy = 0.6

        for iter in range(max_iter):
            conf_mx = self.calculate_conf_mx(mu, worker_annotations_values, worker_annotations_tasks)

            for j in range(n_values):
                prior[j] = np.sum(mu[:, j]) / n_tasks

            self.classifier.update_w(X, mu)

            likelihood = self.calculate_likelihoods(conf_mx, worker_annotations_values, worker_annotations_tasks)
            predictions = self.classifier.get_predictions(X, n_tasks)

            grads = np.linalg.norm((1 - predictions.max(axis=1))[:, None] * X, axis=1) ** 2

            buckets = self.build_gradient_buckets(ground_truth_ids, grads, predictions, n_bucket)
            if confidence_estimator is None:
                # Gonna get it ourself
                estimator = self.get_confidence(buckets, rnd_accuracy)
            else:
                estimator = confidence_estimator
            for i in range(n_tasks):
                confidence = estimator(grads[i])
                l[i] = confidence
            for i in range(n_tasks):
                for j in range(n_values):
                    mu[i, j] = np.log(predictions[i, j] * l[i] + prior[j] * (1 - l[i])) + likelihood[i, j]

            mu = np.exp(mu)
            mu = sklearn.preprocessing.normalize(mu, axis=1, norm='l1')

            l_matrix = np.hstack([l.reshape(-1, 1)] * n_values)
            weights = predictions * l_matrix + np.stack([prior] * n_tasks) * (1 - l_matrix)
            loglike = self.get_loglike(mu, weights, likelihood)

            self.logit_.append(loglike)
            if iter % (max_iter // 5) == 0:
                if axes is not None:
                    x = (iter // (max_iter // 5)) // 2
                    y = (iter // (max_iter // 5)) % 2
                    ds_cls_accuracy = self.ds_classifier_accuracy(np.argmax(mu, axis=1), prior)
                    from benchmark import plot_gradient_buckets
                    plot_gradient_buckets(axes[x, y], buckets, rnd_accuracy, ds_cls_accuracy)
                    axes[x, y].set_title(f'After {iter + 1} epochs')
                print(f'Iter {iter:02}, logit: {loglike:.6f}')
                print(f'Average Raykar weight is {l.mean()}')
            self.mus.append(mu.copy())
            self.cls.append(predictions.copy())
            self.conf_mxs.append(conf_mx.copy())
            self.priors.append(prior.copy())
            self.weights.append(l.copy())
            self.evaluate_classifier(test)

        self.mus = np.array(self.mus)
        self.cls = np.array(self.cls)
        self.conf_mxs = np.array(self.conf_mxs)
        self.priors = np.array(self.priors)
        self.weights = np.array(self.weights)

        self.predictions_ = {t: (
            self.values[np.argmax(mu[i, :])], mu[i], predictions[i], grads[i], likelihood[i], l[i], (l[i], 1 - l[i]), i)
            for
            t, i in self.task_to_id.items()}
        print(f'Average Raykar weight is {l.mean()}')
        self.conf_mx = conf_mx

    def apply_classifier(self, features: np.ndarray) -> Dict[str, str]:
        return self.classifier.apply(features, self.values)

    @classmethod
    def build_gradient_buckets(cls, ground_truth_ids, grads, predictions, n_bucket):
        n_samples = len(grads)
        correct = np.argmax(predictions, axis=1) == ground_truth_ids
        order = np.argsort(grads)
        n_buckets = int(np.ceil(n_samples / n_bucket))
        buckets = []

        for i in range(n_buckets):
            cur_points = order[n_bucket * i: min(n_bucket * (i + 1), n_samples)]
            # print(predictions[cur_points].argmax(axis=1))
            buckets.append(GradientBucket(grads[cur_points], correct[cur_points]))

        return buckets

    def get_ground_truth_ids(self, data):
        gold_dict = {e.task: e.value for e in data.gold()}
        ground_truth_ids = []
        for task in self.tasks:
            ground_truth_ids.append(self.value_to_id[gold_dict[task]])
        return np.array(ground_truth_ids)

    @classmethod
    def get_confidence(cls, buckets: List[GradientBucket], rnd_accuracy: float):
        confidences = []
        right_ends = []
        for i in range(len(buckets)):
            size = buckets[i].size
            confidences.append(binom.sf(size * rnd_accuracy, size, p=buckets[i].accuracy))
            # confidences.append(1 - (np.random.binomial(1, p=buckets[i].accuracy, size=(10000, size)).sum(axis=1) <=
            #                         size * rnd_accuracy).mean())
            right_ends.append(buckets[i].right_end)
        right_ends[-1] += 1

        def get_confidence_bind(x):
            return confidences[bisect.bisect_left(right_ends, x)]

        return get_confidence_bind

    @staticmethod
    def ds_classifier_accuracy(ground_truth, prior):
        accuracy = 0
        for i in range(len(prior)):
            accuracy += (ground_truth == i).mean() * prior[i]

        return accuracy
