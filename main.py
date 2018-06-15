import os
import pandas as pd
import numpy as np
import zipfile
import data
import validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier


def unzip_dataset_files(base_dir):
    for file_name in os.listdir(base_dir):
        if '.zip' in file_name:
            zip_path = os.path.join(base_dir, file_name)
            zip = zipfile.ZipFile(zip_path)
            zip.extractall(base_dir)
            zip.close()
            os.remove(zip_path)


# Carrega bancos de dados e faz processamento da tabela de treinamento e testes
def process_dataframe(dataset_dir, dummies_features):
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

    full_df = data.convert_categorical_features(full_df)
    full_df = data.column_fillna(full_df, data.identify_columns_with_na(full_df), '')

    dummies_features = []
    for feature in data.get_categorical_features(full_df):
        if len(full_df[feature].unique()) > 2 and len(full_df[feature].unique()) <= 30:
            dummies_features.append(feature)
    full_df = pd.get_dummies(full_df, columns=dummies_features)

    train_df = full_df[full_df.dataset == 0]
    del train_df['dataset']

    test_df = full_df[full_df.dataset == 1]
    del test_df['dataset']
    del test_df['TARGET']

    print('Processed full_df memory size %s' % (data.memory_usage(full_df)))
    data.validate_dataframe(full_df, train_df, test_df)

    del full_df
    return train_df, test_df


if __name__ == '__main__':
    dataset_dir = os.path.abspath('./dataset')
    unzip_dataset_files(dataset_dir)

    # Treinamento do modelo
    train_df, test_df = process_dataframe(dataset_dir, None)
    X = train_df.drop(['TARGET', 'SK_ID_CURR'], axis=1).values
    y = train_df['TARGET'].values

    classifier = ExtraTreesClassifier()
    predict, scores, success_items = validation.cross_val_predict_proba(classifier, X, y, None, n_folds=3, verbose=True)
    classifier.fit(X, y)

    '''
    # Preparando base de dados
    train_df['success'] = success_items
    error_rate = (len(train_df[train_df.success == 0]) / len(train_df) * 100)
    print('\tModel error rate: %f' % error_rate)
    train_df.to_csv('error_analysis_train_df_%f.csv' % error_rate)
    '''
    del train_df

    # Output
    X_test = test_df.drop(['SK_ID_CURR'], axis=1).values
    y_pred = classifier.predict_proba(X_test)[:, 1]

    output = pd.DataFrame({'SK_ID_CURR': test_df.SK_ID_CURR, 'TARGET': y_pred})
    output.to_csv('results\\result_%f.csv' % scores.mean(), index=False)
