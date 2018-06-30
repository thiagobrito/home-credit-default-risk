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
    return 'layer_1_model_lgbm_v3'


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
                                   13000, 130000],
                  'is_unbalance': [True, False]}
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
    features_to_drop = ['TARGET', 'TRAIN', 'SK_ID_CURR', 'SK_ID_BUREAU',
                        # removendo features com correlacao 1.0
                        # layer_1_model_lgbm_v3 [ 0.74027767  0.74787884  0.74235477  0.75003905  0.75075121]
                        'POS_MONTHS_BALANCE_SIZE', 'CC_AMT_RECIVABLE_MIN', 'APPROVED_AMT_APPLICATION_MAX',
                        'CC_NAME_CONTRACT_STATUS_Sent proposal_VAR', 'APPROVED_AMT_GOODS_PRICE_MAX',
                        'NEW_ANNUITY_TO_INCOME_RATIO', 'PREV_PRODUCT_COMBINATION_nan_MEAN',
                        'PREV_NAME_CONTRACT_TYPE_XNA_MEAN', 'ANNUITY_INCOME_PERC',
                        'CC_NAME_CONTRACT_STATUS_Refused_SUM', 'CC_NAME_CONTRACT_STATUS_Refused_VAR',
                        'CC_NAME_CONTRACT_STATUS_Refused_MEAN', 'MEAN_DAYS_ENDDATE_DIFF',
                        'CC_NAME_CONTRACT_STATUS_Sent proposal_SUM', 'PREV_NAME_CONTRACT_STATUS_Unused offer_MEAN',
                        'CC_NAME_CONTRACT_STATUS_Sent proposal_MAX', 'POS_COUNT', 'NEW_EMPLOY_TO_BIRTH_RATIO',
                        'PREV_CODE_REJECT_REASON_CLIENT_MEAN', 'CC_AMT_TOTAL_RECEIVABLE_MIN', 'AVG_ENDDATE_FUTURE',
                        'DAYS_EMPLOYED_PERC', 'CC_NAME_CONTRACT_STATUS_Refused_MAX',
                        'CC_NAME_CONTRACT_STATUS_Sent proposal_MEAN',
                        # removendo features com correlacao >= 0.9
                        'CC_NAME_CONTRACT_STATUS_Refused_SUM', 'PREV_AMT_APPLICATION_MEAN', 'APPROVED_AMT_CREDIT_MEAN',
                        'PREV_NAME_SELLER_INDUSTRY_Furniture_MEAN', 'CC_AMT_PAYMENT_CURRENT_MAX', 'AMT_CREDIT',
                        'CC_AMT_RECEIVABLE_PRINCIPAL_SUM', 'AMT_GOODS_PRICE', 'CC_CNT_DRAWINGS_CURRENT_SUM',
                        'APPROVED_RATE_DOWN_PAYMENT_MIN', 'INSTAL_PAYMENT_PERC_MAX',
                        'APPROVED_HOUR_APPR_PROCESS_START_MIN', 'NEW_DOC_IND_AVG', 'CC_SK_DPD_DEF_VAR',
                        'REFUSED_AMT_GOODS_PRICE_MAX', 'PREV_AMT_CREDIT_MAX', 'ELEVATORS_AVG', 'FLOORSMAX_MEDI',
                        'CC_AMT_BALANCE_VAR', 'CC_AMT_PAYMENT_CURRENT_SUM', 'APPROVED_HOUR_APPR_PROCESS_START_MEAN',
                        'REFUSED_AMT_APPLICATION_MEAN', 'BASEMENTAREA_AVG', 'FLOORSMIN_MODE', 'APARTMENTS_MEDI',
                        'OBS_30_CNT_SOCIAL_CIRCLE', 'PREV_CODE_REJECT_REASON_CLIENT_MEAN', 'PREV_DAYS_DECISION_MIN',
                        'YEARS_BEGINEXPLUATATION_AVG', 'APPROVED_AMT_ANNUITY_MEAN', 'MAX_DAYS_ENDDATE_DIFF',
                        'EMERGENCYSTATE_MODE_No', 'REGION_RATING_CLIENT', 'CC_SK_DPD_SUM',
                        'CC_CNT_DRAWINGS_POS_CURRENT_SUM', 'ELEVATORS_MEDI', 'CC_AMT_DRAWINGS_OTHER_CURRENT_MAX',
                        'CC_AMT_TOTAL_RECEIVABLE_MIN', 'CC_AMT_TOTAL_RECEIVABLE_VAR',
                        'CC_NAME_CONTRACT_STATUS_Signed_VAR', 'PREV_AMT_DOWN_PAYMENT_MIN', 'PREV_DAYS_DECISION_MEAN',
                        'NEW_DOC_IND_STD', 'CC_AMT_PAYMENT_TOTAL_CURRENT_VAR', 'CC_SK_DPD_VAR', 'ELEVATORS_MODE',
                        'APARTMENTS_AVG', 'APPROVED_AMT_ANNUITY_MIN', 'INSTAL_AMT_INSTALMENT_MAX', 'CC_SK_DPD_MEAN',
                        'APPROVED_AMT_CREDIT_MIN', 'CC_AMT_RECIVABLE_VAR', 'APPROVED_APP_CREDIT_PERC_MIN',
                        'CC_CNT_DRAWINGS_OTHER_CURRENT_MIN', 'CC_AMT_PAYMENT_TOTAL_CURRENT_MEAN',
                        'PREV_AMT_GOODS_PRICE_MEAN', 'OBS_60_CNT_SOCIAL_CIRCLE',
                        'CC_NAME_CONTRACT_STATUS_Sent proposal_SUM', 'CC_NAME_CONTRACT_STATUS_Sent proposal_MAX',
                        'REFUSED_AMT_GOODS_PRICE_MIN', 'NEW_EMPLOY_TO_BIRTH_RATIO', 'NEW_ANNUITY_TO_INCOME_RATIO',
                        'POS_MONTHS_BALANCE_SIZE', 'PREV_NAME_CONTRACT_STATUS_Unused offer_MEAN',
                        'REFUSED_AMT_GOODS_PRICE_MEAN', 'POS_SK_DPD_DEF_MEAN', 'STD_DAYS_ENDDATE_DIFF',
                        'CC_SK_DPD_MAX', 'PREV_APP_CREDIT_PERC_MEAN', 'PREV_AMT_GOODS_PRICE_MAX',
                        'CC_NAME_CONTRACT_STATUS_Signed_MEAN', 'CC_AMT_BALANCE_MAX',
                        'CC_NAME_CONTRACT_STATUS_Completed_VAR', 'NONLIVINGAPARTMENTS_AVG',
                        'HOUSETYPE_MODE_block of flats', 'CC_MONTHS_BALANCE_VAR', 'APPROVED_APP_CREDIT_PERC_MEAN',
                        'CC_AMT_TOTAL_RECEIVABLE_SUM', 'REFUSED_DAYS_DECISION_MIN', 'PREV_AMT_CREDIT_MIN',
                        'YEARS_BUILD_AVG', 'CC_SK_DPD_DEF_SUM', 'STD_DAYS_DIFF', 'REFUSED_AMT_CREDIT_MAX',
                        'APPROVED_AMT_APPLICATION_MEAN', 'NONLIVINGAREA_MODE', 'LANDAREA_AVG',
                        'CC_AMT_RECEIVABLE_PRINCIPAL_VAR', 'CC_AMT_RECIVABLE_MAX', 'YEARS_BUILD_MEDI',
                        'LIVINGAREA_MEDI', 'REFUSED_RATE_DOWN_PAYMENT_MIN', 'NEW_EXT_SOURCES_MEAN',
                        'ANNUITY_INCOME_PERC', 'PREV_NAME_PORTFOLIO_Cars_MEAN', 'CC_MONTHS_BALANCE_MIN',
                        'CC_MONTHS_BALANCE_MEAN', 'INSTAL_AMT_PAYMENT_SUM', 'REFUSED_HOUR_APPR_PROCESS_START_MEAN',
                        'CC_NAME_CONTRACT_STATUS_Refused_MAX', 'CC_NAME_CONTRACT_STATUS_Sent proposal_MEAN',
                        'CC_AMT_RECIVABLE_SUM', 'LANDAREA_MEDI', 'PREV_HOUR_APPR_PROCESS_START_MEAN',
                        'APPROVED_AMT_APPLICATION_MIN', 'COMMONAREA_MEDI', 'CC_CNT_DRAWINGS_CURRENT_MEAN',
                        'APPROVED_RATE_DOWN_PAYMENT_MAX', 'CC_AMT_CREDIT_LIMIT_ACTUAL_MAX',
                        'CC_AMT_PAYMENT_TOTAL_CURRENT_SUM', 'PREV_NAME_CASH_LOAN_PURPOSE_XNA_MEAN',
                        'PREV_PRODUCT_COMBINATION_nan_MEAN', 'REFUSED_AMT_DOWN_PAYMENT_MIN', 'CC_AMT_BALANCE_MEAN',
                        'INSTAL_AMT_PAYMENT_MEAN', 'NEW_INC_PER_CHLD', 'REFUSED_AMT_CREDIT_MIN',
                        'PREV_AMT_DOWN_PAYMENT_MEAN', 'PREV_NAME_CONTRACT_TYPE_Revolving loans_MEAN', 'ENTRANCES_AVG',
                        'PREV_AMT_ANNUITY_MAX', 'INSTAL_AMT_PAYMENT_MAX', 'CC_CNT_DRAWINGS_POS_CURRENT_MEAN',
                        'PREV_AMT_DOWN_PAYMENT_MAX', 'CC_AMT_PAYMENT_CURRENT_MEAN', 'ENTRANCES_MEDI',
                        'POS_SK_DPD_MEAN', 'INSTAL_AMT_INSTALMENT_MEAN', 'CC_AMT_RECEIVABLE_PRINCIPAL_MAX',
                        'PREV_APP_CREDIT_PERC_MIN', 'MAX_DAYS_DIFF', 'PREV_RATE_DOWN_PAYMENT_MIN',
                        'REFUSED_RATE_DOWN_PAYMENT_MAX', 'CC_NAME_CONTRACT_STATUS_Active_SUM', 'NONLIVINGAREA_MEDI',
                        'LIVINGAPARTMENTS_AVG', 'CC_CNT_DRAWINGS_CURRENT_MAX', 'ORGANIZATION_TYPE_XNA',
                        'PREV_NAME_GOODS_CATEGORY_Clothing and Accessories_MEAN', 'CC_AMT_DRAWINGS_OTHER_CURRENT_MIN',
                        'CC_AMT_TOTAL_RECEIVABLE_MEAN', 'CC_CNT_DRAWINGS_CURRENT_VAR', 'NONLIVINGAREA_AVG',
                        'REFUSED_HOUR_APPR_PROCESS_START_MIN', 'REFUSED_AMT_DOWN_PAYMENT_MAX',
                        'APPROVED_AMT_DOWN_PAYMENT_MIN', 'APPROVED_AMT_DOWN_PAYMENT_MAX', 'BASEMENTAREA_MODE',
                        'TOTALAREA_MODE', 'CC_AMT_BALANCE_MIN', 'LIVINGAPARTMENTS_MODE',
                        'CC_CNT_DRAWINGS_POS_CURRENT_VAR', 'MEAN_DAYS_ENDDATE_DIFF',
                        'CC_NAME_CONTRACT_STATUS_Refused_MEAN', 'PREV_NAME_GOODS_CATEGORY_XNA_MEAN',
                        'REFUSED_AMT_APPLICATION_MAX', 'YEARS_BUILD_MODE', 'PREV_NAME_CONTRACT_TYPE_XNA_MEAN',
                        'APPROVED_AMT_CREDIT_MAX', 'FLOORSMAX_AVG', 'POS_SK_DPD_MAX',
                        'CC_CNT_INSTALMENT_MATURE_CUM_MAX', 'APPROVED_AMT_GOODS_PRICE_MEAN',
                        'CC_AMT_DRAWINGS_OTHER_CURRENT_SUM', 'CC_AMT_RECIVABLE_MEAN', 'CC_MONTHS_BALANCE_SUM',
                        'CC_AMT_BALANCE_SUM', 'CC_AMT_INST_MIN_REGULARITY_VAR', 'CC_AMT_INST_MIN_REGULARITY_MEAN',
                        'NONLIVINGAPARTMENTS_MODE', 'REGION_RATING_CLIENT_W_CITY', 'APPROVED_APP_CREDIT_PERC_MAX',
                        'REFUSED_AMT_DOWN_PAYMENT_MEAN', 'CC_NAME_CONTRACT_STATUS_Sent proposal_VAR',
                        'PREV_AMT_ANNUITY_MEAN', 'LANDAREA_MODE', 'PREV_AMT_ANNUITY_MIN',
                        'APPROVED_AMT_DOWN_PAYMENT_MEAN', 'APARTMENTS_MODE', 'CC_NAME_CONTRACT_STATUS_Refused_VAR',
                        'CC_AMT_RECIVABLE_MIN', 'PREV_CHANNEL_TYPE_Car dealer_MEAN', 'APPROVED_AMT_ANNUITY_MAX',
                        'COMMONAREA_MODE', 'CC_AMT_CREDIT_LIMIT_ACTUAL_MEAN', 'MEAN_DAYS_DIFF', 'DAYS_EMPLOYED_PERC',
                        'LIVINGAPARTMENTS_MEDI', 'INCOME_PER_PERSON', 'CC_CNT_DRAWINGS_POS_CURRENT_MAX',
                        'POS_SK_DPD_DEF_MAX', 'CC_AMT_PAYMENT_TOTAL_CURRENT_MAX', 'CC_CNT_INSTALMENT_MATURE_CUM_VAR',
                        'CC_AMT_INST_MIN_REGULARITY_MAX', 'PREV_NAME_CONTRACT_TYPE_Cash loans_MEAN',
                        'APPROVED_AMT_GOODS_PRICE_MIN', 'REFUSED_AMT_ANNUITY_MEAN', 'APPROVED_DAYS_DECISION_MEAN',
                        'REFUSED_AMT_ANNUITY_MAX', 'COMMONAREA_AVG', 'STD_RISK_SCORE',
                        'CC_NAME_CONTRACT_STATUS_Active_VAR', 'CC_COUNT', 'ENTRANCES_MODE',
                        'APPROVED_AMT_GOODS_PRICE_MAX', 'REFUSED_AMT_CREDIT_MEAN', 'PREV_NAME_PORTFOLIO_Cards_MEAN',
                        'REFUSED_AMT_ANNUITY_MIN', 'CC_AMT_TOTAL_RECEIVABLE_MAX', 'INSTAL_PAYMENT_PERC_SUM',
                        'PREV_APP_CREDIT_PERC_MAX', 'CC_AMT_RECEIVABLE_PRINCIPAL_MIN',
                        'PREV_HOUR_APPR_PROCESS_START_MAX', 'REFUSED_DAYS_DECISION_MAX', 'FLOORSMIN_MEDI',
                        'YEARS_BEGINEXPLUATATION_MEDI', 'CC_AMT_PAYMENT_CURRENT_VAR',
                        'PREV_NAME_GOODS_CATEGORY_Furniture_MEAN', 'REFUSED_DAYS_DECISION_MEAN', 'MAX_RISK_SCORE',
                        'CC_AMT_RECEIVABLE_PRINCIPAL_MEAN', 'FLOORSMIN_AVG', 'YEARS_BEGINEXPLUATATION_MODE',
                        'PREV_AMT_GOODS_PRICE_MIN', 'REFUSED_RATE_DOWN_PAYMENT_MEAN', 'LIVINGAREA_MODE',
                        'CC_CNT_INSTALMENT_MATURE_CUM_SUM', 'APPROVED_RATE_DOWN_PAYMENT_MEAN',
                        'REFUSED_HOUR_APPR_PROCESS_START_MAX', 'PREV_NAME_SELLER_INDUSTRY_XNA_MEAN',
                        'APPROVED_AMT_APPLICATION_MAX', 'LIVINGAREA_AVG', 'PREV_NAME_SELLER_INDUSTRY_Clothing_MEAN',
                        'CC_AMT_DRAWINGS_CURRENT_SUM', 'PREV_RATE_DOWN_PAYMENT_MEAN', 'REFUSED_AMT_APPLICATION_MIN',
                        'APPROVED_DAYS_DECISION_MIN', 'FLOORSMAX_MODE', 'PREV_AMT_APPLICATION_MAX',
                        'BASEMENTAREA_MEDI', 'INSTAL_AMT_INSTALMENT_SUM', 'PREV_HOUR_APPR_PROCESS_START_MIN',
                        'NAME_INCOME_TYPE_Pensioner', 'PREV_RATE_DOWN_PAYMENT_MAX',
                        'PREV_NAME_CONTRACT_TYPE_Consumer loans_MEAN', 'AVG_ENDDATE_FUTURE', 'NEW_SOURCES_PROD',
                        'CC_AMT_INST_MIN_REGULARITY_SUM', 'CC_CNT_INSTALMENT_MATURE_CUM_MEAN',
                        'APPROVED_HOUR_APPR_PROCESS_START_MAX', 'PREV_NAME_PORTFOLIO_POS_MEAN',
                        'NONLIVINGAPARTMENTS_MEDI', 'POS_COUNT', 'PREV_AMT_APPLICATION_MIN', 'PREV_AMT_CREDIT_MEAN',
                        'AMT_INCOME_TOTAL']

    features = list(set(train_df.columns) - set(features_to_drop))

    x_train, y_train = data.extract_values_from_dataframe(train_df, features)
    # params_optimize(x_train, y_train)

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
