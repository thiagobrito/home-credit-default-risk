import pandas as pd
import xgboost as xgb

import data
import validation


def name():
    return 'layer_1_model_xgboost_v1'


def run(train_df, test_df):
    features_to_drop = ['TARGET', 'TRAIN', 'SK_ID_CURR', 'SK_ID_BUREAU', 'SK_ID_PREV', 'index']
    features = list(set(train_df.columns) - set(features_to_drop))

    classifier = xgb.XGBClassifier(base_score=0.5, colsample_bylevel=1, colsample_bytree=0.68,
                                   gamma=0, learning_rate=0.03, max_delta_step=0, max_depth=7,
                                   min_child_weight=1, missing=None, n_estimators=1000, nthread=-1,
                                   objective='binary:logistic', reg_alpha=0, reg_lambda=1,
                                   scale_pos_weight=1, seed=0, silent=True, subsample=0.75)

    train_pred, test_pred, cross_scores = validation.cross_val_predict(classifier, train_df, test_df, features,
                                                                       use_proba=True)

    x_tournament, _ = data.extract_values_from_dataframe(test_df, features)
    return cross_scores, \
           pd.DataFrame(data={'SK_ID_CURR': train_df["SK_ID_CURR"].values, 'TARGET': train_pred}), \
           pd.DataFrame(data={'SK_ID_CURR': test_df["SK_ID_CURR"].values, 'TARGET': test_pred})


if __name__ == '__main__':
    import numpy as np

    np.random.seed(1985)

    train_predict_df, test_predict_df = data.load_dataset()
    scores, train_predict_df, test_predict_df = run(train_predict_df, test_predict_df)

    data.save_submission(train_predict_df, name(), 'train', scores)
    data.save_submission(test_predict_df, name(), 'test', scores)
