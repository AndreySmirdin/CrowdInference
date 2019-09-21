import unittest

from crowd_inference.methods.dawid_skene import DawidSkene
from crowd_inference.methods.majority_vote import MajorityVote
from crowd_inference.truth_inference import NoFeaturesInference
from .data_provider import SimpleGeneratedDataProvider, RelDataProvider, AdultsDataProvider, DataProvider


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
        mv = MajorityVote()
        self._assert_accuracy(self._simple_data, mv, 0.33)
        self._assert_accuracy(self._adults_data, mv, 0.76)
        self._assert_accuracy(self._rel_data, mv, 0.53)

    def test_dawid_skene(self):
        ds = DawidSkene()
        self._assert_accuracy(self._simple_data, ds, 0.33)
        self._assert_accuracy(self._adults_data, ds, 0.76)
        self._assert_accuracy(self._rel_data, ds, 0.61)

    def _assert_accuracy(self, provider: DataProvider, inference: NoFeaturesInference,
                         expected_accuracy: float) -> None:
        accepted = 0
        all_points = 0
        estimates = {}

        inference.fit(provider.labels())
        for estimate in inference.estimate():
            estimates[estimate.task] = estimate.value
        for point in provider.gold():
            if point.task in estimates:
                estimate = estimates[point.task]
                all_points += 1
                if point.value == estimate:
                    accepted += 1

        accuracy = accepted / all_points
        self.assertAlmostEqual(expected_accuracy, accuracy, places=2)


if __name__ == '__main__':
    unittest.main()
