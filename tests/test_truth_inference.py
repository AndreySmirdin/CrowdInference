import unittest
from typing import Iterable

from crowd_inference.methods.majority_vote import MajorityVote
from crowd_inference.methods.dawid_skene import DawidSkene
from crowd_inference.model.estimation import Estimation
from crowd_inference.truth_inference import TruthInference
from .data_provider import SimpleGeneratedDataProvider, RelDataProvider, AdultsDataProvider


class TestTruthInference(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import os
        print(os.path.dirname(os.path.abspath(__file__)))
        cls._simple_data = SimpleGeneratedDataProvider()
        cls._rel_data = RelDataProvider('./resources/datasets/rel/trec-rf10-data.txt')
        cls._adults_data = AdultsDataProvider('./resources/datasets/adults/labels.txt',
                                              './resources/datasets/adults/gold.txt')

    def test_majority_vote(self):
        inference = MajorityVote()

        inference.fit(self._simple_data.labels())
        self.assertAlmostEqual(self._accuracy(self._simple_data.gold(), inference), 0.33, places=2)

        inference.fit(self._adults_data.labels())
        self.assertAlmostEqual(self._accuracy(self._adults_data.gold(), inference), 0.76, places=2)

        inference.fit(self._rel_data.labels())
        self.assertAlmostEqual(self._accuracy(self._rel_data.gold(), inference), 0.53, places=2)

    def test_dawid_skene(self):
        inference = DawidSkene()

        # inference.fit(self._simple_data.labels())
        # print(f"Simple accuracy: {self._accuracy(self._simple_data.gold(), inference)}")

        inference.fit(self._adults_data.labels())
        print(f"Adult accuracy: {self._accuracy(self._adults_data.gold(), inference)}")

        inference.fit(self._rel_data.labels())
        print(f"Rel accuracy: {self._accuracy(self._rel_data.gold(), inference)}")

    @staticmethod
    def _accuracy(gold: Iterable[Estimation], inference: TruthInference) -> float:
        accepted = 0
        all_points = 0
        estimates = {}
        for estimate in inference.estimate():
            estimates[estimate.task] = estimate.value
        for point in gold:
            if point.task in estimates:
                estimate = estimates[point.task]
                all_points += 1
                if point.value == estimate:
                    accepted += 1
        return accepted / all_points


if __name__ == '__main__':
    unittest.main()
