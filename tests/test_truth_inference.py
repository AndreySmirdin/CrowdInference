import unittest
import numpy as np

from crowd_inference.methods.dawid_skene import DawidSkene
from crowd_inference.methods.majority_vote import MajorityVote
from crowd_inference.methods.raykar import Raykar
from crowd_inference.methods.raykar_boosting import RaykarWithBoosting
from crowd_inference.methods.raykar_plus_ds import RaykarPlusDs
from crowd_inference.truth_inference import NoFeaturesInference, TruthInference, WithFeaturesInference
from .data_provider import SimpleGeneratedDataProvider, RelDataProvider, AdultsDataProvider, DataProvider, \
    MusicDataProvider, SentimentDataProvider, IonosphereProvider, MushroomsDataProvider, TolokaDataProvider


class TestTruthInference(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import os
        print(os.path.dirname(os.path.abspath(__file__)))
        np.random.seed(100)
        cls._simple_data = SimpleGeneratedDataProvider()
        flip_probs = [0.1, 0.2, 0.3, 0.5, 0.6]
        annotate_prob = 0.7
        # cls._mushrooms_data = MushroomsDataProvider(resample=False, flip_probs=flip_probs, annotate_prob=annotate_prob)
        cls._rel_data = RelDataProvider('./resources/datasets/rel/trec-rf10-data.txt')
        cls._adults_data = AdultsDataProvider('./resources/datasets/adults/labels.txt',
                                              './resources/datasets/adults/gold.txt')
        cls._music_data = MusicDataProvider()
        cls._sentiment_data = SentimentDataProvider('./resources/datasets/sentiment_polarity/mturk_answers.csv',
                                                    './resources/datasets/sentiment_polarity/polarity_gold_lsa_topics.csv')


        cls._ionosphere_data = IonosphereProvider('./resources/datasets/ionosphere/ionosphere.pickle', resample=False,
                                                 path='./resources/datasets/ionosphere/ionosphere.csv',
                                                 flip_probs=flip_probs,
                                                 annotate_prob=annotate_prob)
        cls._toloka_data = TolokaDataProvider()
        # print(len(cls._ionosphere_data.labels()), len(cls._ionosphere_data.gold()))

    def test_majority_vote(self):
        mv = MajorityVote()
        # self._assert_accuracy(self._simple_data, mv, 0.33)
        # self._assert_accuracy(self._adults_data, mv, 0.76)
        # self._assert_accuracy(self._rel_data, mv, 0.53)
        # self._assert_accuracy(self._ionosphere_data, mv, 0.80)
        self._assert_accuracy(self._sentiment_data, mv, 0.88)
        # self._assert_accuracy(self._mushrooms_data, mv, 0.76)
        # self._assert_accuracy(self._toloka_data, mv, 0.83)
        # self._assert_accuracy(self._music_data, mv, 0.79)

    def test_dawid_skene(self):
        ds = DawidSkene()
        # self._assert_accuracy(self._simple_data, ds, 0.33)
        # self._assert_accuracy(self._adults_data, ds, 0.76)
        # self._assert_accuracy(self._rel_data, ds, 0.61)
        # self._assert_accuracy(self._ionosphere_data, ds, 0.85)
        # self._assert_accuracy(self._sentiment_data, ds, 0.91)
        # self._assert_accuracy(self._mushrooms_data, ds, 0.81)
        self._assert_accuracy(self._toloka_data, ds, 0.86)
        # self._assert_accuracy(self._music_data, ds, 0.75)

    def test_raykar(self):
        raykar = Raykar()
        self._assert_accuracy(self._mushrooms_data, raykar, 0.91)
        self._assert_accuracy(self._ionosphere_data, raykar, 0.92)
        # print(self._get_accuracy(self._sentiment_data, raykar))

    def test_raykar_boosting(self):
        raykar = RaykarWithBoosting()
        self._assert_accuracy(self._mushrooms_data, raykar, 0.91)
        self._assert_accuracy(self._ionosphere_data, raykar, 0.92)

    def test_ionosphere(self):
        raykar = Raykar()
        self._get_accuracy(self._ionosphere_data, raykar)

        self._get_classifier_accuracy(raykar, self._ionosphere_data)

    def test_raykar_plus_ds(self):
        raykar_plus_ds = RaykarPlusDs()
        self._assert_accuracy(self._ionosphere_data, raykar_plus_ds, 0.71)

        self._get_classifier_accuracy(raykar_plus_ds, self._ionosphere_data)

    def test_compare_on_ionosphere(self):
        results = []
        logits = []
        for method in [DawidSkene(), Raykar(), RaykarPlusDs()]:
            accuracy = self._get_accuracy(self._ionosphere_data, method)
            results.append((method.__str__(), accuracy))
            logits.append((method.__str__(), method.logit_))

        print(results)
        print(logits)

    def _get_classifier_accuracy(self, inference: WithFeaturesInference, data_provider: DataProvider):
        classification = inference.apply_classifier(data_provider.features())

        correct = 0
        for item in data_provider.gold():
            if classification[item.task] == item.value:
                correct += 1
        print(f'Classifier accuracy is {correct / len(classification)}')

    def _get_accuracy(self, provider: DataProvider, inference: TruthInference):
        accepted = 0
        all_points = 0
        estimates = {}

        if isinstance(inference, WithFeaturesInference):
            inference.fit(provider.labels(), provider.features())
        elif isinstance(inference, NoFeaturesInference):
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
        print(f'Label accuracy is {accuracy}')

        return accuracy

    def _assert_accuracy(self, provider: DataProvider, inference: TruthInference,
                         expected_accuracy: float) -> None:
        self.assertAlmostEqual(expected_accuracy, self._get_accuracy(provider, inference), places=2)


if __name__ == '__main__':
    unittest.main()
