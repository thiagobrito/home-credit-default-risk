import pandas as pd
import category_encoders as ce


def transform(train_df, test_df):
    full_df = pd.concat([train_df, test_df])
    full_df.fillna(-1, inplace=True)
    full_df.ix[full_df.TARGET == -1, 'test'] = 1
    full_df.ix[full_df.TARGET != -1, 'test'] = 0

    encoder = ce.TargetEncoder()
    full_df = encoder.fit_transform(full_df, full_df['TARGET'])

    train_df, test_df = full_df[full_df.test == 0], full_df[full_df.test == 1]
    del train_df['test']
    del test_df['test']
    return train_df, test_df


def one_hot_encoder(df, nan_as_category=True):
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns=categorical_columns, dummy_na=nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns
