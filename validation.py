import os
import numpy as np
import json
import datetime
import operator
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
import paths


def cross_validation_index(X, y, n_folds, random_state):
    return StratifiedKFold(n_splits=n_folds, random_state=random_state, shuffle=False).split(X, y)


def cross_val_predict(classifier, train_df, test_df, features, n_splits=5, random_state=0, use_proba=True):
    train_pred = np.zeros(train_df.shape[0])
    test_pred = np.zeros(test_df.shape[0])
    cross_scores = np.empty(0)

    folds = StratifiedKFold(n_splits=n_splits, shuffle=False, random_state=random_state)
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[features], train_df['TARGET'])):
        train_x, train_y = train_df[features].iloc[train_idx], train_df['TARGET'].iloc[train_idx]
        valid_x, valid_y = train_df[features].iloc[valid_idx], train_df['TARGET'].iloc[valid_idx]

        classifier.fit(train_x, train_y, eval_set=[(train_x, train_y), (valid_x, valid_y)],
                       eval_metric='auc', early_stopping_rounds=200)

        if use_proba:
            train_pred[valid_idx] = classifier.predict_proba(valid_x)[:, 1]
            test_pred += classifier.predict_proba(test_df[features])[:, 1] / folds.n_splits
        else:
            train_pred[valid_idx] = classifier.predict(valid_x)
            test_pred += classifier.predict(test_df[features]) / folds.n_splits

        score = metrics.roc_auc_score(valid_y, train_pred[valid_idx])
        cross_scores = np.append(cross_scores, score)

        print('Fold %2d AUC : %.6f' % (n_fold + 1, score))

    print('Final cross scores: %s' % cross_scores)
    return train_pred, test_pred, cross_scores


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
