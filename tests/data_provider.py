import csv
import os
import pickle

import numpy as np
import pandas as pd
from abc import abstractmethod
from typing import Iterable, Dict, List, Optional, Tuple

import sklearn
from sklearn import preprocessing
from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import MinMaxScaler

from crowd_inference.model.annotation import Annotation
from crowd_inference.model.estimation import Estimation


class DataProvider:

    @abstractmethod
    def labels(self) -> Iterable[Annotation]:
        pass

    @abstractmethod
    def gold(self) -> Iterable[Estimation]:
        pass


def group_data(features, gold):
    id_list = []
    feature_list = []
    gold_list = []

    for g in gold:
        id_list.append(g.task)
        gold_list.append(g.value)
        feature_list.append(features[g.task])
    return np.array(id_list), np.array(feature_list, dtype='float'), np.array(gold_list)


class SimpleGeneratedDataProvider(DataProvider):
    def __init__(self):
        self._simple_labels = []
        self._simple_labels.append(Annotation('1', '1', 'T'))
        self._simple_labels.append(Annotation('1', '2', 'T'))
        self._simple_labels.append(Annotation('1', '3', 'T'))
        self._simple_labels.append(Annotation('2', '1', 'F'))
        self._simple_labels.append(Annotation('2', '2', 'F'))
        self._simple_labels.append(Annotation('2', '3', 'F'))
        self._simple_labels.append(Annotation('3', '1', 'T'))
        self._simple_labels.append(Annotation('3', '2', 'T'))
        self._simple_labels.append(Annotation('3', '3', 'F'))

        self._simple_gold = []
        self._simple_gold.append(Estimation('1', 'F'))
        self._simple_gold.append(Estimation('2', 'T'))
        self._simple_gold.append(Estimation('3', 'T'))

    def labels(self) -> Iterable[Annotation]:
        return self._simple_labels

    def gold(self) -> Iterable[Estimation]:
        return self._simple_gold


class RelDataProvider(DataProvider):
    def __init__(self, path: str):
        self._rel_labels = []
        self._rel_gold = set()
        with open(path, newline='') as csvfile:
            file_reader = csv.reader(csvfile, delimiter='\t')
            next(file_reader)
            for row in file_reader:
                self._rel_labels.append(Annotation(row[1], row[0] + '#' + row[2], row[4]))
                gold = row[3]
                if gold != '-1':
                    self._rel_gold.add(Estimation(row[0] + '#' + row[2], gold))

    def labels(self) -> Iterable[Annotation]:
        return self._rel_labels

    def gold(self) -> Iterable[Estimation]:
        return self._rel_gold


class AdultsDataProvider(DataProvider):
    def __init__(self, labels_path: str, gold_path: str):
        self._adult_labels = []
        self._adult_gold = []
        with open(labels_path, newline='') as csvfile:
            file_reader = csv.reader(csvfile, delimiter='\t')
            for row in file_reader:
                self._adult_labels.append(Annotation(row[0], row[1], row[2]))
        with open(gold_path, newline='') as csvfile:
            file_reader = csv.reader(csvfile, delimiter='\t')
            for row in file_reader:
                self._adult_gold.append(Estimation(row[0], row[1]))

    def labels(self) -> Iterable[Annotation]:
        return self._adult_labels

    def gold(self) -> Iterable[Estimation]:
        return self._adult_gold


class MusicDataProvider(DataProvider):
    ANNOTATIONS_PATH = './resources/datasets/music_genre/music_genre_mturk.csv'
    GOLD_PATH = './resources/datasets/music_genre/music_genre_gold.csv'
    TEST_PATH = './resources/datasets/music_genre/music_genre_test.csv'

    def __init__(self):
        self._music_labels = []
        self._music_gold = []
        self._features = {}
        data = pd.read_csv(self.ANNOTATIONS_PATH)
        for _, row in data.iterrows():
            self._music_labels.append(Annotation(row['annotator'], row['id'], row['class']))

        for id_, group in data.groupby(['id']):
            gold = id_.split('.')[0]
            self._music_gold.append(Estimation(id_, gold))

        def get_features(x):
            x = x[1:-1]
            return x

        scaler, n_features = self.read_features(get_features)

        test = pd.read_csv(self.TEST_PATH)
        n = len(test)
        self.X = np.zeros((n, n_features))
        self.y = []

        for i, row in test.iterrows():
            cur_features = get_features(row.values).reshape(1, -1)
            cur_features = scaler.transform(cur_features).reshape(-1)
            self.X[i] = np.concatenate([cur_features, [1]])
            self.y.append(row.values[-1])

        self.y = np.array(self.y)

    def labels(self) -> Iterable[Annotation]:
        return self._music_labels

    def gold(self) -> Iterable[Estimation]:
        return self._music_gold

    def features(self) -> Dict[str, np.ndarray]:
        return self._features

    def test(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.X, self.y

    def read_features(self, get_features):
        gold = pd.read_csv(self.GOLD_PATH)
        features, ids = [], []
        for _, row in gold.iterrows():
            features.append(get_features(row.values))
            ids.append(row['id'])

        features = np.array(features)
        scaler = preprocessing.MinMaxScaler().fit(features)
        features = scaler.transform(features)
        features = np.hstack([features, np.ones((len(features), 1))])
        for i, f in zip(ids, features):
            self._features[i] = f

        return scaler, features.shape[1]


class IonosphereProvider(DataProvider):
    def __init__(self, save_path, resample: bool = False, path: Optional[str] = None,
                 flip_probs: Optional[List[float]] = None,
                 annotate_prob: Optional[float] = None):
        self._labels = []
        self._gold = []
        self._features = {}

        if not resample:
            with open(save_path, 'rb') as f:
                self._labels, self._gold, self._features = pickle.load(f)
            return

        data = pd.read_csv(path)
        for i, row in data.iterrows():
            for _ in range(1):
                has_annotation = np.random.binomial(1, annotate_prob, len(flip_probs))
                while has_annotation.sum() == 0:
                    has_annotation = np.random.binomial(1, annotate_prob, len(flip_probs))
                for j, p in enumerate(flip_probs):
                    if has_annotation[j]:
                        if np.random.binomial(1, 1 - p):
                            label = row[-1]
                        else:
                            label = 'b' if row[-1] == 'g' else 'g'
                        self._labels.append(Annotation(str(j), str(i), label))
        print(len(self._labels))
        for i, row in data.iterrows():
            to_array = row.values
            to_array[1] += 1  # Fix zero feature
            self._gold.append(Estimation(str(i), to_array[-1]))
            self._features[str(i)] = to_array[:-1]

        with open(save_path, 'wb') as f:
            pickle.dump((self._labels, self._gold, self._features), f)

    def labels(self) -> Iterable[Annotation]:
        return self._labels

    def gold(self) -> Iterable[Estimation]:
        return self._gold

    def features(self) -> Dict[str, np.ndarray]:
        return self._features


class MushroomsDataProvider(DataProvider):
    PATH = './resources/datasets/mushrooms/mushrooms.txt'
    SAVE_PATH = './resources/datasets/mushrooms/mushrooms.pickle'

    def __init__(self, resample: bool = False,
                 flip_probs: Optional[List[float]] = None,
                 annotate_prob: Optional[float] = None, test_train_split=0.8):
        self._labels = []
        self._gold = []
        self._features = {}

        if not resample:
            with open(self.SAVE_PATH, 'rb') as f:
                self._labels, self._gold, self._features, self.testX, self.testY = pickle.load(f)
            return

        X, y = load_svmlight_file(self.PATH)
        train_size = int(len(y) * test_train_split)
        test_size = len(y) - train_size
        print((y == 1).sum())
        print((y == 2).sum())
        X = np.asarray(X.todense())
        X = np.hstack([X, np.ones((X.shape[0], 1))]).astype('float')
        y = np.array(list(map(lambda x: str(int(x)), y)))

        for i in range(train_size):
            for _ in range(1):
                has_annotation = np.random.binomial(1, annotate_prob, len(flip_probs))
                while has_annotation.sum() == 0:
                    has_annotation = np.random.binomial(1, annotate_prob, len(flip_probs))

                for j, p in enumerate(flip_probs):
                    if has_annotation[j]:
                        if np.random.binomial(1, 1 - p):
                            label = y[i]
                        else:
                            label = '1' if y[i] == '2' else '2'
                        self._labels.append(Annotation(str(j), str(i), label))

            self._gold.append(Estimation(str(i), y[i]))
            self._features[str(i)] = X[i]
            # self._features[str(i)] += np.random.randn(len(self._features[str(i)])) * 0.5
        self.testX = X[-test_size:]
        # self.testX += np.random.randn(*self.testX.shape) * 0.5
        self.testY = y[-test_size:]

        with open(self.SAVE_PATH, 'wb') as f:
            pickle.dump((self._labels, self._gold, self._features, self.testX, self.testY), f)

    def labels(self) -> Iterable[Annotation]:
        return self._labels

    def gold(self) -> Iterable[Estimation]:
        return self._gold

    def features(self) -> Dict[str, np.ndarray]:
        return self._features

    def test(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.testX, self.testY


class SentimentDataProvider(DataProvider):
    TEST_PATH = './resources/datasets/sentiment_polarity/polarity_test_lsa_topics.csv'

    def __init__(self, labels_path: str, gold_path: str):
        self._sentiment_labels = []
        self._sentiment_gold = []
        self._features = {}
        labels = pd.read_csv(labels_path)
        for _, row in labels.iterrows():
            self._sentiment_labels.append(Annotation(row['WorkerId'], row['Input.id'], row['Answer.sent']))

        gold = pd.read_csv(gold_path)

        def get_features(x):
            return x[1:-1]

        features = []
        for _, row in gold.iterrows():
            self._sentiment_gold.append(Estimation(row['id'], row['class']))
            features.append(get_features(row.values))

        features = np.array(features)
        print(features.shape)
        scaler = MinMaxScaler().fit(features)
        # features = scaler.transform(features)
        features = np.hstack([features, np.ones((len(features), 1))])
        for i, e in enumerate(self._sentiment_gold):
            self._features[e.task] = features[i]

        n_features = features.shape[1]

        test = pd.read_csv(self.TEST_PATH)
        n = len(test)
        self.X = np.zeros((n, n_features - 1))
        self.y = []

        for i, row in test.iterrows():
            features = get_features(row.values)
            self.X[i] = features
            self.y.append(row.values[-1])
        # self.X = scaler.transform(self.X)
        self.X = np.hstack([self.X, np.ones((len(self.X), 1))])

        self.y = np.array(self.y)

    def labels(self) -> Iterable[Annotation]:
        return self._sentiment_labels

    def gold(self) -> Iterable[Estimation]:
        return self._sentiment_gold

    def features(self) -> Dict[str, np.ndarray]:
        return self._features

    def test(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.X, self.y


class TolokaDataProvider(DataProvider):
    DIR = './resources/datasets/TlkAggFtrs'
    FEATURES_PATH = os.path.join(DIR, 'features.tsv')
    GOLD_PATH = os.path.join(DIR, 'golden_labels.tsv')
    CROWD_PATH = os.path.join(DIR, 'crowd_labels.tsv')

    def __init__(self):
        self._toloka_labels = []
        self._toloka_gold = []
        self._features = {}
        self.X, self.y = [], []

        tasks = set()
        test_labels = {}
        with open(self.GOLD_PATH, 'r') as f:
            for line in f:
                words = line.split()
                task, label = words[0], words[1]
                tasks.add(task)
                if self._is_test_task(task):
                    test_labels[task] = label
                else:
                    self._toloka_gold.append(Estimation(task, label))

        with open(self.CROWD_PATH, 'r') as f:
            for line in f:
                words = line.split()
                if words[1] in tasks:
                    if int(words[0][-1]) % 5 == 0 and not self._is_test_task(words[1]):
                        self._toloka_labels.append(Annotation(words[0], words[1], words[2]))

        with open(self.FEATURES_PATH, 'r') as f:
            for line in f:
                split = line.split()
                task = split[0]
                if task in tasks:
                    features = list(map(float, split[1:]))
                    if self._is_test_task(task):
                        self.X.append(np.concatenate([features, [1]]))
                        self.y.append(test_labels[task])
                    else:
                        self._features[task] = np.concatenate([features, [1]])
        self.X = np.array(self.X)
        self.y = np.array(self.y)

    def labels(self) -> Iterable[Annotation]:
        return self._toloka_labels

    def gold(self) -> Iterable[Estimation]:
        return self._toloka_gold

    def features(self) -> Dict[str, np.ndarray]:
        return self._features

    def test(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.X, self.y

    def _is_test_task(self, name):
        n = int(name[1:])
        return n % 8 == 0


class CovTypeDataProvider(DataProvider):
    SAVE_PATH = './resources/datasets/cov_type.pickle'

    def __init__(self, resample: bool = False,
                 flip_probs: Optional[List[float]] = None,
                 annotate_prob: Optional[float] = None, test_train_split=0.8, each_nth=100):
        self._labels = []
        self._gold = []
        self._features = {}

        if not resample:
            with open(self.SAVE_PATH, 'rb') as f:
                self._labels, self._gold, self._features, self.testX, self.testY = pickle.load(f)
            return

        ds = sklearn.datasets.fetch_covtype()
        X, y = ds['data'][::each_nth], ds['target'][::each_nth]
        X = np.hstack([X, np.ones((X.shape[0], 1))]).astype('float')

        a = 2
        b = 1
        cls2 = y == b
        cls1 = np.argwhere(y == a)[:cls2.sum()].reshape(-1)
        y = np.concatenate([y[cls1], y[cls2]])
        order = np.random.permutation(len(y))
        X = MinMaxScaler().fit_transform(X)
        X = np.vstack([X[cls1], X[cls2]])[order]
        y = y[order]
        y = np.array(list(map(str, y)))

        train_size = int(len(y) * test_train_split)
        test_size = len(y) - train_size
        print((y == str(a)).sum())
        print((y == str(b)).sum())

        for i in range(train_size):
            for _ in range(1):
                has_annotation = np.random.binomial(1, annotate_prob, len(flip_probs))
                while has_annotation.sum() == 0:
                    has_annotation = np.random.binomial(1, annotate_prob, len(flip_probs))

                for j, p in enumerate(flip_probs):
                    if has_annotation[j]:
                        if np.random.binomial(1, 1 - p):
                            label = y[i]
                        else:
                            label = '1' if y[i] == str(a) else '2'
                        self._labels.append(Annotation(str(j), str(i), label))

            self._gold.append(Estimation(str(i), y[i]))
            self._features[str(i)] = X[i]
        self.testX = X[-test_size:]
        self.testY = y[-test_size:]

        with open(self.SAVE_PATH, 'wb') as f:
            pickle.dump((self._labels, self._gold, self._features, self.testX, self.testY), f)

    def labels(self) -> Iterable[Annotation]:
        return self._labels

    def gold(self) -> Iterable[Estimation]:
        return self._gold

    def features(self) -> Dict[str, np.ndarray]:
        return self._features

    def test(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.testX, self.testY
