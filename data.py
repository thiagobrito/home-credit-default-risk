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


def load_dataset():
    dataset_dir = os.path.join(os.path.dirname(__file__), 'dataset')
    train_df_file_path = os.path.join(dataset_dir, 'application_train.csv')
    test_df_file_path = os.path.join(dataset_dir, 'application_test.csv')

    train_df = pd.read_csv(train_df_file_path)
    test_df = pd.read_csv(test_df_file_path)

    train_df['dataset'] = 0
    test_df['dataset'] = 1
    test_df['TARGET'] = 0

    full_df = pd.concat([train_df, test_df])
    del train_df
    del test_df

    full_df = convert_categorical_features(full_df)

    bureau_processed_df = pd.read_csv(os.path.join(dataset_dir, 'bureau_processed.csv'))
    full_df = full_df.merge(right=bureau_processed_df, on=['SK_ID_CURR'], how='left')

    train_df = full_df[full_df.dataset == 0]
    del train_df['dataset']

    test_df = full_df[full_df.dataset == 1]
    del test_df['dataset']
    del test_df['TARGET']

    print('Processed full_df memory size %s' % (memory_usage(full_df)))
    #validate_dataframe(full_df, train_df, test_df)

    del full_df
    return train_df, test_df


def save_submission(df, model_name, scores):
    df.to_csv(paths.make_results_path(model_name, np.median(scores)), index=False)
