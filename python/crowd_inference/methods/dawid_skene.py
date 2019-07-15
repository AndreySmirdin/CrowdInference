from typing import List, Iterable

from crowd_inference.model.annotation import Annotation
from crowd_inference.truth_inference import NoFeaturesInference


class DawidSkene(NoFeaturesInference):
    def fit(self, annotations: Iterable[Annotation]):
        pass

    def estimation(self, task: str) -> str:
        pass
