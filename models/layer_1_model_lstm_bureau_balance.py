import pandas as pd

from keras.models import *
from keras.layers import *
from sklearn.preprocessing import StandardScaler

import validation
import data


def lstm_model(number_of_features):
    model = Sequential()
    # model.add(LSTM(64, input_shape=(1, 100), return_sequences=True))
    model.add(LSTM(32, input_shape=(1, number_of_features)))
    # model.add(LSTM(32))
    model.add(Dense(1, activation='sigmoid'))

    # model.add(Dense(60, input_dim=98, kernel_initializer='normal', activation='relu'))
    # model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def name():
    return 'layer_1_model_lstm_bureau_balance'


def run(train_df, test_df):
    bureau_balance_df = pd.read_csv('../dataset/bureau_balance_processed.csv')
    bureau_balance_df = bureau_balance_df.dropna(subset=['SK_ID_CURR'])

    features_to_drop = ['TARGET', 'TRAIN', 'SK_ID_CURR', 'SK_ID_BUREAU']
    features_to_scale = list(set(bureau_balance_df.columns) - set(features_to_drop))
    number_of_features = len(features_to_scale)

    # Escala os dados para ficar tudo padronizado e aumentar eficiencia da rede neural
    bureau_balance_df[features_to_scale] = StandardScaler().fit_transform(bureau_balance_df[features_to_scale].values)

    x = bureau_balance_df[bureau_balance_df.TRAIN == 1].drop(features_to_drop, axis=1).values
    x = x.reshape(x.shape[0], 1, x.shape[1])

    x_tournament = bureau_balance_df[bureau_balance_df.TRAIN == 0].drop(features_to_drop, axis=1).values
    x_tournament = x_tournament.reshape(x_tournament.shape[0], 1, x_tournament.shape[1])

    y = bureau_balance_df[bureau_balance_df.TRAIN == 1]['TARGET'].values

    model = lstm_model(number_of_features)
    train_predict, cross_scores, success_items = validation.cross_val_predict_proba(model, x, y, None,
                                                                                    use_proba=False, verbose=True,
                                                                                    n_folds=2)
    bureau_balance_df.loc[bureau_balance_df.TRAIN == 1, 'predict'] = train_predict
    bureau_balance_df.loc[bureau_balance_df.TRAIN == 0, 'predict'] = model.predict(x_tournament)
    bureau_balance_df[['SK_ID_CURR', 'SK_ID_BUREAU', 'predict']].to_csv('../dataset/bureau_balance_predict.csv',
                                                                        index=False)

    train_df = train_df['SK_ID_CURR']
    train_df = train_df.merge(bureau_balance_df[['SK_ID_CURR', 'predict']], on=['SK_ID_CURR'], how='left')

    test_df = test_df['SK_ID_CURR']
    test_df = test_df.merge(bureau_balance_df[['SK_ID_CURR', 'predict']], on=['SK_ID_CURR'], how='left')
    return cross_scores, train_df, test_df


if __name__ == '__main__':
    import numpy as np

    np.random.seed(1985)

    train_df, test_df = data.load_dataset()
    scores, train_predict_df, test_predict_df = run(train_df, test_df)

    data.save_submission(train_predict_df, name(), 'train', scores)
    data.save_submission(test_predict_df, name(), 'test', scores)
