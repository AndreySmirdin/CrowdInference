from collections import defaultdict, Counter
from typing import Iterable

from crowd_inference.model.annotation import Annotation
from crowd_inference.model.estimation import Estimation
from crowd_inference.truth_inference import NoFeaturesInference


class MajorityVote(NoFeaturesInference):
    def __init__(self):
        self.aggregations = {}

    def fit(self, annotations: Iterable[Annotation]):
        self.aggregations = defaultdict(lambda: Counter())
        for point in annotations:
            self.aggregations[point.task][point.value] += 1

    def estimate(self) -> Iterable[Estimation]:
        result = []
        for task, votes in self.aggregations.items():
            common = votes.most_common(1)
            result.append(Estimation(task, common[0][0]))
        return result
