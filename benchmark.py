import numpy as np
import pandas as pd

import random

import crowd_inference.methods.dawid_skene as ds
import crowd_inference.methods.raykar as r
import crowd_inference.methods.raykar_plus_ds as rds
import crowd_inference.methods.classifier as cls

from crowd_inference.truth_inference import NoFeaturesInference, TruthInference, WithFeaturesInference
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression, LogisticRegression
import matplotlib.pyplot as plt
import tests.data_provider as data
from tqdm.auto import tqdm
from typing import Iterable, Dict, List, Optional, Tuple


def features2np(provider: data.DataProvider) -> Tuple[np.ndarray, np.ndarray]:
    features = provider.features()
    n_features = len(features[list(features.keys())[0]])
    n_tasks = len(features)
    X = np.zeros((n_tasks, n_features))
    y = []
    for i, estimation in enumerate(provider.gold()):
        X[i] = features[estimation.task]
        y.append(estimation.value)

    return X, np.array(y)


def get_classifier_accuracy(inference: WithFeaturesInference, data_provider: data.DataProvider):
    X, y = features2np(data_provider)
    classification = inference.apply_classifier(X)
    accuracy = accuracy_score(y, classification)
    print(f'Classifier train accuracy is {accuracy}')

    if isinstance(data_provider, data.MusicDataProvider) \
            or isinstance(data_provider, data.SentimentDataProvider) or isinstance(data_provider,
                                                                                   data.MushroomsDataProvider):
        X, y = data_provider.test()
        classification = inference.apply_classifier(X)
        test_accuracy = accuracy_score(y, classification)
        print(f'Classifier test accuracy is {test_accuracy}')


def get_accuracy(provider: data.DataProvider, inference: TruthInference, max_iter: int, confidence_estimator,
                 lr: float):
    correct = []
    incorrect = []
    accepted = 0
    all_points = 0
    estimates = {}

    if isinstance(inference, ds.DawidSkene):
        inference.fit(provider.labels(), max_iter=max_iter)
    elif isinstance(inference, r.Raykar):
        inference.fit(provider.labels(), provider.features(), max_iter=max_iter, lr=lr)
        get_classifier_accuracy(inference, provider)
    else:
        inference.fit(provider.labels(), provider.features(), max_iter=max_iter,
                      confidence_estimator=confidence_estimator, lr=lr)
        get_classifier_accuracy(inference, provider)

    for estimate in inference.estimate():
        estimates[estimate.task] = estimate.value
    for point in provider.gold():
        if point.task in estimates:
            estimate = estimates[point.task]
            all_points += 1
            if point.value == estimate:
                accepted += 1
                correct.append(point)
            else:
                incorrect.append(point)

    accuracy = accepted / all_points
    print(f'Label accuracy is {accuracy}')

    return accuracy, {e.task for e in correct}, {e.task for e in incorrect}


def compare_methods(provider, max_iter=15, confidence_estimator=None, lr=0.1):
    results = []
    points_results = []
    methods = [ds.DawidSkene(),
               r.Raykar(), rds.RaykarPlusDs(),
               #                rds.RaykarPlusDs(binary=True)
               ]
    points = []

    for method in methods:
        accuracy, correct, incorrect = get_accuracy(provider, method, max_iter, confidence_estimator, lr)
        results.append((method.__str__(), accuracy))
        plt.plot(method.logit_)
        points_results.append((correct, incorrect))
        print('-' * 50)
        task, mu, classifier, likelihood_rds, conf_mx, index, grads = [], [], [], [], [], [], []
        for k, v in method.predictions_.items():
            task.append(k)
            mu.append(np.round(v[1], 3))
            if isinstance(method, WithFeaturesInference):
                classifier.append(np.round(v[2], 3))
                grads.append(np.round(v[3], 5))
                conf_mx.append(np.round(v[4], 3))
            if isinstance(method, rds.RaykarPlusDs):
                #                 method_rds.append("R" if v[3] else "DS")
                likelihood_rds.append(np.round(v[6], 3))
                index.append(v[7])

        columns = {'task': task, 'mu': mu}
        if len(classifier):
            columns['classifier'] = classifier
            columns['grad'] = grads
            columns['conf_mx'] = conf_mx
        if len(likelihood_rds):
            columns['likelihood_rds'] = likelihood_rds
            columns['index'] = index

        points.append(pd.DataFrame(columns))

    print(len(points))
    points_aggregated = points[1].merge(points[2], on='task', suffixes=('_r', '_rds'))
    points_aggregated = points_aggregated.merge(points[0], on='task')
    points_aggregated.rename(columns={'mu': 'mu_ds'}, inplace=True)

    print(results)

    plt.legend(list(map(str, methods)))

    # Get points advantages
    advantages = []
    for i in range(len(methods)):
        advantages.append([])
        for j in range(len(methods)):
            advantages[i].append(points_results[i][0] - points_results[j][0])

    return methods, points_aggregated, advantages


def print_conf(methods):
    for m in methods:
        print(m)
        print(m.conf_mx[:, 0, 0])
        print(m.conf_mx[:, 1, 1])


def shuffle_features(data):
    features_list = []
    for k in data._features.keys():
        features_list.append(data._features[k])
    random.shuffle(features_list)
    for k, f in zip(data._features.keys(), features_list):
        data._features[k] = f


def plots_for_point(points, data, methods, k):
    plt.figure(figsize=(10, 6))

    task = points[points.index == k].task.values[0]
    for est in data.labels():
        if est.task == task:
            print(est)

    plt.plot(methods[2].mus[:, k, 0])
    plt.plot(methods[1].mus[:, k, 0])
    plt.plot(methods[0].mus[:, k, 0])

    plt.plot(methods[2].cls[:, k, 0])
    plt.plot(methods[1].cls[:, k, 0])

    plt.plot(methods[2].weights[:, k])

    plt.legend(['RDS', 'R', 'DS', 'Classifier RDS', 'Classifier R',
                'Raykar weight'])


def build_grad_hist(data, methods, points, name, n_bucket=100):
    gold_dict = {e.task: e.value for e in data.gold()}

    fig = plt.figure(figsize=(12, 8))
    ax = fig.subplots(nrows=2, ncols=1)
    print(f'Number of data points: {len(points)}')

    buckets = []
    confidences = []

    wrong, correct = [], []

    for method in [1]:
        result = []
        buckets.append([])
        for _, p in points.iterrows():
            flipped = False
            #             if  method == 2:
            #                 clazz = np.argmax(p.classifier_rds)
            #                 grad = np.abs(p.grad_rds)
            #             elif method == 1:
            clazz = np.argmax(p.classifier_r)
            grad = np.abs(p.grad_r)
            flipped = np.argmax(p.mu_r) != np.argmax(p.conf_mx_r)

            if np.argmax(p.classifier_r) == methods[method].value_to_id[gold_dict[p.task]]:
                if flipped:
                    correct.append(p)
                result.append([grad, True, flipped, False])
            else:
                if flipped:
                    wrong.append(p)
                result.append([grad, False, False, flipped])

        result = np.array(sorted(result))

        n = int(np.ceil(len(points) / n_bucket))
        xs = []
        widths = []
        heights_accuracy = []
        heights_flipped_good, heights_flipped_bad = [], []
        for i in range(n):
            cur_points = result[n_bucket * i: min(n_bucket * (i + 1), len(points))]
            xs.append((cur_points[0, 0] + cur_points[-1, 0]) * 0.5)
            widths.append((cur_points[-1, 0] - cur_points[0, 0]))
            heights_accuracy.append(cur_points[:, 1].sum() / len(cur_points))
            heights_flipped_good.append(cur_points[:, 2].sum() / len(cur_points))
            heights_flipped_bad.append(cur_points[:, 3].sum() / len(cur_points))

            buckets[-1].append(cur_points[:, 0])

        xs = np.array(xs)
        widths = np.array(widths)

        ax[method - 1].bar(xs, height=heights_accuracy, width=widths)
        #         print(heights_flipped)

        flipped_widths = widths / 2
        ax[1].bar(xs - flipped_widths / 2, height=heights_flipped_good, width=flipped_widths)
        ax[1].bar(xs + flipped_widths / 2, height=heights_flipped_bad, width=flipped_widths)

        ax[method - 1].set_title(name + " Точность работы классификатора в зависимости от градиента")
        ax[method].set_title(name + " Классификатор изменил самый вероятный класс")
        ax[method].legend(['Изменил на правильный', 'Изменил на неправильный'])

        confidences.append(heights_accuracy)

    return buckets, confidences, wrong, correct


def get_confidence(buckets, confidences):
    def get_confidence_binded(x):
        x = np.abs(x)
        for i in range(len(confidences) - 1):
            if x < buckets[i][-1]:
                return confidences[i]
        return confidences[-1]

    return get_confidence_binded


def plot_flips(correct, wrong, name):
    c = list(map(lambda x: x[name].max(), correct))
    w = list(map(lambda x: x[name].max(), wrong))
    plt.hist([c, w])
    plt.legend(['correct', 'wrong'])

    print(len(correct), len(wrong))


