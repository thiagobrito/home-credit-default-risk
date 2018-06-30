import pandas as pd
import lightgbm as lgb

import validation
import data
import paths

from lgbm_util import LgbmAdapter
from models.learning_rate import *

from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.model_selection import train_test_split


def name():
    return 'layer_1_model_lgbm_v2'


# https://www.kaggle.com/mlisovyi/lightgbm-hyperparameter-optimisation-lb-0-761
def params_optimize(x_train, y_train):
    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.20, stratify=y_train)
    fit_params = {"early_stopping_rounds": 30,
                  "eval_metric": 'auc',
                  "eval_set": [(x_test, y_test)],
                  'eval_names': ['valid'],
                  # 'callbacks': [lgb.reset_parameter(learning_rate=learning_rate_010_decay_power_099)],
                  'verbose': 100,
                  'categorical_feature': 'auto'}

    param_test = {'num_leaves': sp_randint(6, 50),
                  'min_child_samples': sp_randint(100, 500),
                  'min_child_weight': [1e-5, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4],
                  'subsample': sp_uniform(loc=0.2, scale=0.8),
                  'colsample_bytree': sp_uniform(loc=0.4, scale=0.6),
                  'reg_alpha': [0, 1e-1, 1, 2, 5, 7, 10, 50, 100],
                  'reg_lambda': [0, 1e-1, 1, 5, 10, 20, 50, 100],
                  'n_estimators': [1000, 10000,
                                   2000, 20000,
                                   3000, 30000,
                                   5000, 50000,
                                   7000, 70000,
                                   13000, 130000]}

    hp_points_to_test = 200
    clf = lgb.LGBMClassifier(max_depth=-1, silent=True, metric='None', n_jobs=4, n_estimators=5000)
    gs = RandomizedSearchCV(
        estimator=clf, param_distributions=param_test,
        n_iter=hp_points_to_test,
        scoring='roc_auc',
        cv=3,
        refit=True,
        verbose=True)

    gs.fit(x_train, y_train, **fit_params)
    print('Best score reached: {} with params: {} '.format(gs.best_score_, gs.best_params_))


def run(train_df, test_df):
    features_to_drop = ['TARGET', 'TRAIN', 'SK_ID_CURR', 'SK_ID_BUREAU']
    features = list(set(train_df.columns) - set(features_to_drop))

    x_train, y_train = data.extract_values_from_dataframe(train_df, features)
    #params_optimize(x_train, y_train)

    dataset = lgb.Dataset(x_train, y_train)

    params = {'n_estimators': 50000, 'learning_rate': 0.1, 'num_leaves': 20, 'colsample_bytree': 0.64331178360329422,
              'subsample': 0.89456134209978089, 'max_depth': 8, 'reg_alpha': 10, 'reg_lambda': 10,
              'min_split_gain': 0.0222415, 'min_child_weight': 1e-05, 'objective': 'binary', 'is_unbalance': True,
              'min_child_samples': 224}

    # RandomGridSearch best params 0.7662859436419786
    # {'colsample_bytree': 0.64331178360329422, 'min_child_samples': 224, 'min_child_weight': 1e-05, 'num_leaves': 20, 'reg_alpha': 10, 'reg_lambda': 10, 'subsample': 0.89456134209978089}

    model = LgbmAdapter(params, dataset, features, 20, learning_rate_010_decay_power_099)
    train_predict, cross_scores, success_items = validation.cross_val_predict_proba(model, x_train, y_train,
                                                                                    features, use_proba=False)
    print(name(), cross_scores)

    x_tournament, _ = data.extract_values_from_dataframe(test_df, features)
    return cross_scores, \
           pd.DataFrame(data={'SK_ID_CURR': train_df["SK_ID_CURR"].values, 'TARGET': train_predict}), \
           pd.DataFrame(data={'SK_ID_CURR': test_df["SK_ID_CURR"].values, 'TARGET': model.predict(x_tournament)})


if __name__ == '__main__':
    import numpy as np

    np.random.seed(1985)

    train_predict_df, test_predict_df = data.load_dataset()
    scores, train_predict_df, test_predict_df = run(train_predict_df, test_predict_df)

    data.save_submission(train_predict_df, name(), 'train', scores)
    data.save_submission(test_predict_df, name(), 'test', scores)
