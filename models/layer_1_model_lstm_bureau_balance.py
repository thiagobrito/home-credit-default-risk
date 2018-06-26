import pandas as pd

from keras.models import *
from keras.layers import *

import validation
import data


def lstm_model():
    model = Sequential()
    model.add(LSTM(64, input_shape=(1, 99), return_sequences=True))
    model.add(LSTM(32))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def name():
    return 'layer_1_model_lstm_bureau_balance'


def run(train_df, test_df):
    bureau_balance_df = pd.read_csv('../dataset/bureau_balance_processed.csv')
    bureau_balance_df = bureau_balance_df.dropna(subset=['SK_ID_CURR'])

    x = bureau_balance_df[bureau_balance_df.TRAIN == 1].drop(['TARGET', 'TRAIN'], axis=1).values
    x = x.reshape(x.shape[0], 1, x.shape[1])
    y = bureau_balance_df[bureau_balance_df.TRAIN == 1]['TARGET'].values

    model = lstm_model()
    train_predict, cross_scores, success_items = validation.cross_val_predict_proba(model, x, y, None, verbose=True)

    x_tournament = bureau_balance_df[bureau_balance_df.TRAIN == 0].drop(['TARGET'], axis=1).values
    return cross_scores, \
           pd.DataFrame(data={'SK_ID_CURR': train_df["SK_ID_CURR"].values, 'TARGET': train_predict}), \
           pd.DataFrame(data={'SK_ID_CURR': test_df["SK_ID_CURR"].values, 'TARGET': model.predict(x_tournament)})


if __name__ == '__main__':
    import numpy as np

    np.random.seed(1985)

    train_df, test_df = data.load_dataset()
    scores, train_predict_df, test_predict_df = run(train_df, test_df)

    data.save_submission(train_predict_df, name(), 'train', scores)
    data.save_submission(test_predict_df, name(), 'test', scores)
