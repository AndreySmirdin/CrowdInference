from abc import abstractmethod
from typing import Iterable

from crowd_inference.model.annotation import Annotation
from crowd_inference.model.estimation import Estimation


class TruthInference:
    @abstractmethod
    def estimate(self) -> Iterable[Estimation]:
        pass


class NoFeaturesInference(TruthInference):
    @abstractmethod
    def fit(self, annotations: Iterable[Annotation]):
        pass
