import os
import pandas as pd
import numpy as np

import paths


def get_categorical_features(df):
    return df.select_dtypes('object').columns.tolist()


def convert_categorical_features(df):
    for column in get_categorical_features(df):
        print(column)
        df.loc[:, column] = df[column].astype('category').cat.codes
    return df


def get_dummies(df, categorical_columns=None):
    if categorical_columns is None:
        categorical_columns = get_categorical_features(df)

    for column in categorical_columns:
        df = pd.concat([df, pd.get_dummies(df[column], prefix=column)], axis=1)
    return df


def column_fillna(df, column, value):
    df[column] = df[column].fillna(df[column].mean())
    return df


def identify_columns_with_na(df):
    return [column for column in list(df.columns) if len(df.loc[df[column].isnull()]) > 0]


def validate_dataframe(df, train_df, test_df):
    assert len(identify_columns_with_na(df)) == 0, 'NaN values found (%s)' % identify_columns_with_na(df)


def memory_usage(pandas_obj):
    if isinstance(pandas_obj, pd.DataFrame):
        usage_b = pandas_obj.memory_usage(deep=True).sum()
    else:
        usage_b = pandas_obj.memory_usage(deep=True)
    usage_mb = usage_b / 1024 ** 2
    return "{:03.2f} MB".format(usage_mb)


def extract_values_from_dataframe(df, features):
    if 'TARGET' in df:
        return df[features].values, df['TARGET'].values
    return df[features].values, None


def load_dataset(save_correlation=False):
    print('Loading data...')
    df = pd.read_csv(paths.make_dataset_path('processed_application.csv'))

    bureau = pd.read_csv(paths.make_dataset_path('processed_bureau.csv'))
    df = df.join(bureau.set_index('SK_ID_CURR'), how='left', on='SK_ID_CURR')

    credit_card_balance_df = pd.read_csv(paths.make_dataset_path('processed_credit_card_balance.csv'))
    df = df.join(credit_card_balance_df, how='left', on='SK_ID_CURR')

    installments_payments_df = pd.read_csv(paths.make_dataset_path('processed_installments_payments.csv'))
    df = df.join(installments_payments_df, how='left', on='SK_ID_CURR')

    pos_cash_df = pd.read_csv(paths.make_dataset_path('processed_pos_cash.csv'))
    df = df.join(pos_cash_df, how='left', on='SK_ID_CURR')

    previous_applications_df = pd.read_csv(paths.make_dataset_path('processed_previous_applications.csv'))
    df = df.join(previous_applications_df, how='left', on='SK_ID_CURR')
    print('Done.')

    if save_correlation:
        print('Save correlation...')
        df.corr().to_csv(paths.make_dataset_path('correlations.csv'), index=False)
        print('Done.')

    return df[df['TARGET'].notnull()], df[df['TARGET'].isnull()]


def save_submission(df, model_name, type, scores):
    df.to_csv(paths.make_results_path(model_name, type, np.median(scores)), index=False)


if __name__ == '__main__':
    load_dataset(save_correlation=True)
