import data
import pandas as pd
import validation

from keras.models import *
from keras.layers import *


def lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(32, input_shape=(1, 99)))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def name():
    return 'layer_1_model_lstm_bureau_balance'


def run(train_df, test_df):
    bureau_balance_df = pd.read_csv('../dataset/bureau_balance_processed.csv')

    x = bureau_balance_df.drop(['TARGET'], axis=1).values
    y = bureau_balance_df['TARGET'].values

    #x = x.reshape(1, x.shape[0], x.shape[1])
    x = x.reshape(x.shape[0], 1, x.shape[1])
    #y = y.reshape(y.shape[0], 1)
    print(x.shape, y.shape)
    print(len(x), len(y))

    model = lstm_model(x.shape)
    model_history = model.fit(x, y, validation_split=0.2, shuffle=True, epochs=100, batch_size=1500, verbose=1)
    model_history


if __name__ == '__main__':
    import numpy as np

    np.random.seed(1985)

    #train_df, test_df = data.load_dataset()
    scores, train_df, test_df = run(None, None)

    #data.save_submission(train_df, name(), 'train', scores)
    #data.save_submission(test_df, name(), 'test', scores)
