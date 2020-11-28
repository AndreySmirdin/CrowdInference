import csv
import pickle

import numpy as np
import pandas as pd
from abc import abstractmethod
from typing import Iterable, Dict, List, Optional, Tuple

from crowd_inference.model.annotation import Annotation
from crowd_inference.model.estimation import Estimation


class DataProvider:

    @abstractmethod
    def labels(self) -> Iterable[Annotation]:
        pass

    @abstractmethod
    def gold(self) -> Iterable[Estimation]:
        pass


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
            return x[1:-1]
        
        gold = pd.read_csv(self.GOLD_PATH)
        n_features = None
        for _, row in gold.iterrows():
            features = get_features(row.values)
            self._features[row['id']] = np.concatenate([features, [1]])
            n_features = len(features) + 1
            
        test = pd.read_csv(self.TEST_PATH)
        n = len(test)
        self.X = np.zeros((n, n_features))
        self.y = []
        
        for i, row in test.iterrows():
            features = get_features(row.values)
            self.X[i] = np.concatenate([features, [1]])
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
            return x[1:-1:3]
        
        n_features = None
        for _, row in gold.iterrows():
            self._sentiment_gold.append(Estimation(row['id'], row['class']))
            features = get_features(row.values)
            self._features[row['id']] = features
            n_features = len(features)
        
        test = pd.read_csv(self.TEST_PATH)
        n = len(test)
        self.X = np.zeros((n, n_features))
        self.y = []
        
        for i, row in test.iterrows():
            features = get_features(row.values)
            self.X[i] = features
            self.y.append(row.values[-1])
            
        self.y = np.array(self.y)

    def labels(self) -> Iterable[Annotation]:
        return self._sentiment_labels

    def gold(self) -> Iterable[Estimation]:
        return self._sentiment_gold

    def features(self) -> Dict[str, np.ndarray]:
        return self._features
    
    def test(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.X, self.y

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
