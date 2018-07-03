import pandas as pd
import lightgbm as lgb

import validation
import data
import paths

from lgbm_util import LgbmAdapter
import cat_features

sorted_features = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'OWN_CAR_AGE', 'AVERAGE_LOAN_TYPE',
                   'AVG_ENDDATE_FUTURE', 'OVERDUE_DEBT_RATIO', 'AVG_CREDITDAYS_PROLONGED', 'BUREAU_LOAN_COUNT',
                   'BUREAU_LOAN_TYPES', 'SUM_RISK_SCORE', 'STD_DAYS_DIFF', 'STD_DAYS_ENDDATE_DIFF', 'STD_RISK_SCORE',
                   'ORGANIZATION_TYPE', 'CREDIT_ENDDATE_PERCENTAGE', 'APARTMENTS_AVG', 'APARTMENTS_MODE',
                   'APARTMENTS_MEDI', 'CODE_GENDER', 'HOUR_APPR_PROCESS_START', 'COMMONAREA_AVG',
                   'NONLIVINGAPARTMENTS_AVG', 'NONLIVINGAREA_AVG', 'COMMONAREA_MODE', 'NONLIVINGAPARTMENTS_MODE',
                   'NONLIVINGAREA_MODE', 'COMMONAREA_MEDI', 'NONLIVINGAPARTMENTS_MEDI', 'NONLIVINGAREA_MEDI',
                   'FONDKAPREMONT_MODE', 'HOUSETYPE_MODE', 'TOTALAREA_MODE', 'TOTAL_CUSTOMER_DEBT',
                   'TOTAL_CUSTOMER_CREDIT', 'TOTAL_CUSTOMER_OVERDUE', 'CNT_CHILDREN', 'CNT_FAM_MEMBERS',
                   'ENTRANCES_AVG', 'ENTRANCES_MODE', 'ENTRANCES_MEDI', 'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY',
                   'AMT_GOODS_PRICE', 'EMERGENCYSTATE_MODE', 'AMT_REQ_CREDIT_BUREAU_HOUR', 'AMT_REQ_CREDIT_BUREAU_DAY',
                   'AMT_REQ_CREDIT_BUREAU_WEEK', 'AMT_REQ_CREDIT_BUREAU_MON', 'AMT_REQ_CREDIT_BUREAU_QRT',
                   'AMT_REQ_CREDIT_BUREAU_YEAR', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'FLAG_MOBIL', 'FLAG_EMP_PHONE',
                   'FLAG_WORK_PHONE', 'FLAG_CONT_MOBILE', 'FLAG_PHONE', 'FLAG_EMAIL', 'ELEVATORS_AVG', 'FLOORSMAX_AVG',
                   'FLOORSMIN_AVG', 'ELEVATORS_MODE', 'FLOORSMAX_MODE', 'FLOORSMIN_MODE', 'ELEVATORS_MEDI',
                   'FLOORSMAX_MEDI',
                   'FLOORSMIN_MEDI', 'FLAG_DOCUMENT_2', 'FLAG_DOCUMENT_3', 'FLAG_DOCUMENT_4', 'FLAG_DOCUMENT_5',
                   'FLAG_DOCUMENT_6', 'FLAG_DOCUMENT_7', 'FLAG_DOCUMENT_8', 'FLAG_DOCUMENT_9', 'FLAG_DOCUMENT_10',
                   'FLAG_DOCUMENT_11', 'FLAG_DOCUMENT_12', 'FLAG_DOCUMENT_13', 'FLAG_DOCUMENT_14', 'FLAG_DOCUMENT_15',
                   'FLAG_DOCUMENT_16', 'FLAG_DOCUMENT_17', 'FLAG_DOCUMENT_18', 'FLAG_DOCUMENT_19', 'FLAG_DOCUMENT_20',
                   'FLAG_DOCUMENT_21', 'LIVE_REGION_NOT_WORK_REGION', 'LIVE_CITY_NOT_WORK_CITY', 'LIVINGAPARTMENTS_AVG',
                   'LIVINGAREA_AVG', 'LIVINGAPARTMENTS_MODE', 'LIVINGAREA_MODE', 'LIVINGAPARTMENTS_MEDI',
                   'LIVINGAREA_MEDI',
                   'MIN_DAYS_DIFF', 'MIN_DAYS_ENDDATE_DIFF', 'MIN_RISK_SCORE', 'REGION_POPULATION_RELATIVE',
                   'REGION_RATING_CLIENT', 'REGION_RATING_CLIENT_W_CITY', 'WEEKDAY_APPR_PROCESS_START',
                   'REG_REGION_NOT_LIVE_REGION', 'REG_REGION_NOT_WORK_REGION', 'REG_CITY_NOT_LIVE_CITY',
                   'REG_CITY_NOT_WORK_CITY', 'YEARS_BEGINEXPLUATATION_AVG', 'YEARS_BUILD_AVG',
                   'YEARS_BEGINEXPLUATATION_MODE',
                   'YEARS_BUILD_MODE', 'YEARS_BEGINEXPLUATATION_MEDI', 'YEARS_BUILD_MEDI', 'DEF_30_CNT_SOCIAL_CIRCLE',
                   'DEF_60_CNT_SOCIAL_CIRCLE', 'MEAN_DAYS_DIFF', 'MEAN_DAYS_ENDDATE_DIFF', 'DEBT_CREDIT_RATIO',
                   'MEAN_RISK_SCORE', 'OCCUPATION_TYPE', 'ACTIVE_LOANS_PERCENTAGE', 'OBS_30_CNT_SOCIAL_CIRCLE',
                   'OBS_60_CNT_SOCIAL_CIRCLE', 'NAME_CONTRACT_TYPE', 'NAME_TYPE_SUITE', 'NAME_INCOME_TYPE',
                   'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'DAYS_BIRTH', 'DAYS_EMPLOYED',
                   'DAYS_REGISTRATION', 'DAYS_ID_PUBLISH', 'BASEMENTAREA_AVG', 'LANDAREA_AVG', 'BASEMENTAREA_MODE',
                   'LANDAREA_MODE', 'BASEMENTAREA_MEDI', 'LANDAREA_MEDI', 'WALLSMATERIAL_MODE',
                   'DAYS_LAST_PHONE_CHANGE',
                   'MAX_DAYS_DIFF', 'MAX_DAYS_ENDDATE_DIFF', 'MAX_RISK_SCORE']


def name():
    return 'layer_1_model_lgbm'


def run(train_df, test_df):
    features_to_drop = ['TARGET', 'TRAIN', 'SK_ID_CURR', 'SK_ID_BUREAU']
    features = list(set(train_df.columns) - set(features_to_drop))

    x_train, y_train = data.extract_values_from_dataframe(train_df, features)

    dataset = lgb.Dataset(x_train, y_train)
    params = {'boosting_type': 'gbdt', 'class_weight': None, 'colsample_bytree': 0.7, 'learning_rate': 0.1,
              'max_depth': 3, 'min_child_samples': 20, 'min_child_weight': 0.001, 'min_split_gain': 0.0,
              'n_estimators': 150, 'n_jobs': -1, 'num_leaves': 2,
              'objective': 'binary',
              'reg_alpha': 0,
              'reg_lambda': 0, 'silent': True, 'subsample': 0.8, 'subsample_for_bin': 200000, 'subsample_freq': 1,
              'nthread': 3, 'bagging_fraction': 0.5, 'bagging_freq': 5, 'feature_fraction': 0.2, 'is_unbalance': True,
              'metric': {'auc'},
              'verbose': 0}

    # Use small learning_rate with large num_iterations
    params['num_iterations'] = 5000

    classifier = LgbmAdapter(params, dataset, features, 20)
    train_pred, test_pred, cross_scores = validation.cross_val_predict(classifier, train_df, test_df, features,
                                                                       use_proba=False)

    print(name(), cross_scores)
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
