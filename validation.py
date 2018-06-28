import os
import numpy as np
import json
import datetime
import operator
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
import paths


def cross_validation_index(X, y, n_folds, random_state):
    return StratifiedKFold(n_splits=n_folds, random_state=random_state).split(X, y)


def cross_val_predict_proba(classifier, X, y, features, n_folds=5, random_state=0, verbose=True, use_proba=True):
    cross_scores = np.empty(0)
    predict = np.empty(y.shape)
    success_items = np.zeros(y.shape)

    i = 0
    for index_train, index_test in cross_validation_index(X, y, n_folds, random_state):
        start_time = datetime.datetime.now()

        if verbose:
            print('\tcv=%d' % i, end=' ', flush=True)

        classifier.fit(X[index_train], y[index_train]) #, epochs=2)
        if use_proba:
            predict[index_test] = classifier.predict_proba(X[index_test])[:, 1]
        else:
            predict[index_test] = classifier.predict(X[index_test])
        score = metrics.roc_auc_score(y[index_test], predict[index_test])

        # Identificando os itens que foram sucesso em pelo menos uma passagem
        success_items[index_test] = np.logical_or(success_items[index_test], predict[index_test] == y[index_test])

        cross_scores = np.append(cross_scores, score)
        if verbose:
            print('time: %s | score=%f | importance=%s' % (datetime.datetime.now() - start_time, score,
                                                           show_classificator_data(classifier, features)))
        i += 1
    if verbose:
        print('Final score: %f' % metrics.roc_auc_score(y, predict))
    return predict, cross_scores, success_items


def show_classificator_data(algoritm, features):
    if features:
        importance_dict = {}

        for i, column in enumerate(features):
            try:
                importance_dict[column] = algoritm.feature_importances_[i]
            except:
                importance_dict[column] = list(algoritm.feature_importances_.tolist())[i]

        sorted_importance_dict = list(sorted(iter(importance_dict), key=operator.itemgetter(1), reverse=True))
        return sorted_importance_dict
    return '[]'


def calculate_score(y, predict):
    return metrics.roc_auc_score(y, predict)
