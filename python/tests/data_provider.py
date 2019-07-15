import csv
from abc import abstractmethod
from typing import Iterable

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
