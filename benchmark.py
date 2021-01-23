import numpy as np
import pandas as pd

import random

import sklearn

import crowd_inference.methods.dawid_skene as ds
import crowd_inference.methods.raykar as r
import crowd_inference.methods.raykar_boosting as rb
import crowd_inference.methods.raykar_plus_ds as rds
import crowd_inference.methods.classifier as cls

from crowd_inference.truth_inference import NoFeaturesInference, TruthInference, WithFeaturesInference
from sklearn.metrics import accuracy_score, log_loss
from sklearn.linear_model import LinearRegression, LogisticRegression
import matplotlib.pyplot as plt
import tests.data_provider as data
from statsmodels.stats.proportion import proportion_confint
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
    # print(classification[:10], y[:10])
    accuracy = accuracy_score(y, classification)
    print(f'Classifier train accuracy is {accuracy}')

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
    elif isinstance(inference, r.Raykar) or isinstance(inference, rb.RaykarWithBoosting):
        inference.fit(provider.labels(), provider.features(), max_iter=max_iter, lr=lr, test=provider.test())
        get_classifier_accuracy(inference, provider)
    else:
        inference.fit(provider.labels(), provider.features(), max_iter=max_iter,
                      confidence_estimator=confidence_estimator, lr=lr, test=provider.test())
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
    methods = [
        ds.DawidSkene(),
        r.Raykar(),
        rds.RaykarPlusDs(),
        # rds.RaykarPlusDs(binary=True),
        # rb.RaykarWithBoosting()
    ]
    points = []

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    test_stats = []

    for method in methods:
        accuracy, correct, incorrect = get_accuracy(provider, method, max_iter, confidence_estimator, lr)
        results.append((method.__str__(), accuracy))
        axes[0].plot(method.logit_)
        if method.__str__() != 'DS':
            axes[1].plot(method.losses)
            axes[1].plot(method.accuracies)
            test_stats.append(str(method) + ' loss')
            test_stats.append(str(method) + ' accuracy')
        points_results.append((correct, incorrect))
        print('-' * 50)
        task, mu, classifier, likelihood_rds, conf_mx, index, grads = [], [], [], [], [], [], []
        for k, v in method.predictions_.items():
            task.append(k)
            mu.append(np.round(v[1], 3))
            conf_mx.append(np.round(v[4], 3))
            if isinstance(method, WithFeaturesInference):
                classifier.append(np.round(v[2], 3))
                grads.append(np.round(v[3], 5))

            if isinstance(method, rds.RaykarPlusDs):
                likelihood_rds.append(np.round(v[6], 3))
                index.append(v[7])

        columns = {'task': task, 'mu' + method.suffix(): mu, 'conf_mx' + method.suffix(): conf_mx}
        if len(classifier):
            columns['classifier' + method.suffix()] = classifier
            columns['grad' + method.suffix()] = grads
            columns['conf_mx' + method.suffix()] = conf_mx
        if len(likelihood_rds):
            columns['likelihood' + method.suffix()] = likelihood_rds
            columns['index'] = index

        points.append(pd.DataFrame(columns))

    print(len(points))
    points_aggregated = points[0]
    for i in range(1, len(points)):
        points_aggregated = points_aggregated.merge(points[i], on='task')

    print(results)

    axes[0].legend(list(map(str, methods)))
    axes[1].legend(test_stats)

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


def get_rnd_cls_accuracy(data, method):
    values = np.array(method.values)
    gold = np.array([g.value for g in data.gold()])

    accuracies = []
    for value in values:
        accuracies.append((value == gold).mean())
    accuracies = np.array(accuracies)
    print(accuracies)
    print((accuracies ** 2).sum())
    return (accuracies ** 2).sum()


def build_grad_hist(data, methods, points, name, n_bucket=100, train=True):
    gold_dict = {e.task: e.value for e in data.gold()}

    fig = plt.figure(figsize=(11, 13))
    ax = fig.subplots(nrows=2, ncols=1)

    buckets = []
    confidences = []
    for method in [1]:
        result = []
        if train:
            for _, p in points.iterrows():
                grad = p.grad_r[-1]
                flipped = np.argmax(p.mu_r) != np.argmax(p.conf_mx_r)

                if np.argmax(p.classifier_r) == methods[method].value_to_id[gold_dict[p.task]]:
                    result.append([grad, True, flipped, False])
                else:
                    result.append([grad, False, False, flipped])

        else:
            X, y = data.test()
            predictions = methods[1].classifier.get_predictions(X, len(X))
            grads = np.linalg.norm((1 - predictions.max(axis=1))[:, None] * X, axis=1) ** 2
            for i in range(len(X)):
                if y[i] == methods[method].values[np.argmax(predictions[i])]:
                    result.append([grads[i], True, False, False])
                else:
                    result.append([grads[i], False, False, False])
            points = X

        print(f'Number of data points: {len(points)}')

        result = np.array(sorted(result))

        n = int(np.ceil(len(points) / n_bucket))
        xs = []
        widths = []
        successes, totals = [], []
        heights_flipped_good, heights_flipped_bad = [], []
        for i in range(n):
            cur_points = result[n_bucket * i: min(n_bucket * (i + 1), len(points))]
            xs.append((cur_points[0, 0] + cur_points[-1, 0]) * 0.5)
            widths.append((cur_points[-1, 0] - cur_points[0, 0]))
            successes.append(cur_points[:, 1].sum())
            totals.append(len(cur_points))
            heights_flipped_good.append(cur_points[:, 2].sum() / len(cur_points))
            heights_flipped_bad.append(cur_points[:, 3].sum() / len(cur_points))

            buckets.append(cur_points[:, 0])

        xs = np.array(xs)
        widths = np.array(widths)
        successes, totals = np.array(successes), np.array(totals)
        heights_accuracy = successes / totals
        conf_intervals = []
        for s, t in zip(successes, totals):
            low, up = proportion_confint(s, t)
            conf_intervals.append((up - low) * 0.5)
        rnd_accuracy = get_rnd_cls_accuracy(data, methods[0])
        ax[0].bar(xs, height=heights_accuracy, yerr=conf_intervals, width=widths)
        ax[0].set_xlabel('Градиент')
        ax[0].set_ylabel('Точность классификатора')
        ax[0].plot(xs, [rnd_accuracy] * len(widths), 'red')
        ax[0].legend(['Random classifier'])
        print(heights_accuracy)
        print(widths)
        print(result[:5, 0])
        flipped_widths = widths / 2
        ax[1].bar(xs - flipped_widths / 2, height=heights_flipped_good, width=flipped_widths)
        ax[1].bar(xs + flipped_widths / 2, height=heights_flipped_bad, width=flipped_widths)

        ax[method - 1].set_title("Точность работы классификатора в зависимости от градиента")
        ax[method].set_title(name + " Классификатор изменил самый вероятный класс")
        ax[method].legend(['Изменил на правильный', 'Изменил на неправильный'])

        confidences.append(heights_accuracy)

    return buckets, heights_accuracy, result[-1, 0], rnd_accuracy


def get_confidence(buckets, confidences, rnd_accuracy):
    for i in range(len(confidences)):
        n = len(buckets[i])
        confidences[i] = \
            1 - (np.random.binomial(1, p=confidences[i], size=(10000, n)).sum(axis=1) <= n * rnd_accuracy).mean()

    def get_confidence_binded(x):
        x = np.abs(x)
        for i in range(len(confidences) - 1):
            if x < buckets[i][-1]:
                return confidences[i]
        return confidences[-1]

    return get_confidence_binded


def plot_flips(correct, wrong, name, dataset_name, l1, l2):
    c = list(map(lambda x: x.max(), correct[name]))
    w = list(map(lambda x: x.max(), wrong[name]))
    plt.hist([c, w])
    plt.legend([l1, l2])
    plt.title(dataset_name + ' ' + name)

    print(len(correct), len(wrong))


def boosting_classifiers_distr(pts1, pts2, name, metric='minmax'):
    print(len(pts1), len(pts2))

    def get_scores(pts):
        scores = []
        for p in pts.classifier_rb:
            scores.append(p[:, 0])
        scores = np.array(scores)
        return scores

    scores1 = get_scores(pts1)
    scores2 = get_scores(pts2)
    if metric == 'minmax':
        plt.hist([scores1.max(axis=1) - scores1.min(axis=1), scores2.max(axis=1) - scores2.min(axis=1)])
    elif metric == 'var':
        plt.hist([np.var(scores1, axis=1), np.var(scores2, axis=1)])
    else:
        raise ValueError('Unknown metric')
    plt.title(name + ' ' + metric + ' distribution')
    plt.legend(['correct', 'wrong'])


def plot_all_gradients(pts1, pts2, name):
    for g in pts1.grad_r:
        plt.plot(g, color='red', linewidth=1)
    for g in pts2.grad_r:
        plt.plot(g, color='blue', linewidth=1)


def run_mv_classifier(dataset, n_classes=2, iters=100, lr=0.1, C=1, hard=False):
    X_train, y_train = features2np(dataset)
    c = cls.Classifier(n_features=X_train.shape[1], n_classes=n_classes, lr=lr, C=C)

    inference = TruthInference()
    inference.get_annotation_parameters(dataset.labels())
    mu = inference.get_majority_vote_probs(dataset.labels())

    X = X_train
    Xs = X.T.dot(X)

    def add_accuracy(X, labels, accuracies, losses):
        y_pred = c.get_predictions(X, len(X))
        losses.append(log_loss(labels, y_pred))
        accuracies.append(accuracy_score(labels, inference.values[np.argmax(y_pred, axis=1)]))

    accuracies_train, accuracies_test, losses_train, losses_test = [], [], [], []

    if hard:
        max_pos = mu.argmax(axis=1)
        mu[:, :] = 0
        for i, label in enumerate(inference.values):
            mu[y_train == label, i] = 1
        # for i in range(mu.shape[0]):
        #     mu[i, max_pos[i]] = 1

    for _ in tqdm(range(iters)):
        c.update_w(X, Xs, mu)
        add_accuracy(dataset.test()[0], dataset.test()[1], accuracies_test, losses_test)
        add_accuracy(X_train, y_train, accuracies_train, losses_train)
    _, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].plot(accuracies_test)
    axes[0].plot(accuracies_train)

    axes[1].plot(losses_test)
    axes[1].plot(losses_train)

    plt.legend(['Test', 'Train'])

    return accuracies_test[-1], max(accuracies_test)


def mv_hard(dataset, C=1):
    X_train, y_train = features2np(dataset)

    inference = TruthInference()
    inference.get_annotation_parameters(dataset.labels())
    mu = inference.get_majority_vote_probs(dataset.labels())
    print(mu[:15])
    reg = LogisticRegression(fit_intercept=False, C=C).fit(X_train, y_train)

    X_test, y_test = dataset.test()
    print(y_test)
    print(reg.predict(X_test))
    # print(accuracy_score(y_train, inference.values[reg.predict(X_train)]))
    # return accuracy_score(y_test, inference.values[reg.predict(X_test)])
    print(accuracy_score(y_train, reg.predict(X_train)))
    return accuracy_score(y_test, reg.predict(X_test))
