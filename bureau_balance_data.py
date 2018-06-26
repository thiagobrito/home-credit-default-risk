import pandas as pd

if __name__ == '__main__':
    bureau_balance_df = pd.read_csv('dataset/bureau_balance.csv')

    bureau_balance_df['MONTHS_BALANCE'] = bureau_balance_df['MONTHS_BALANCE'] * -1
    bureau_balance_df['STATUS'] = bureau_balance_df['STATUS'].map({'C': -1, 'X': -2, '0': 0, '1': 1, '2': 2})

    new_df = bureau_balance_df.pivot(index='SK_ID_BUREAU', columns='MONTHS_BALANCE', values='STATUS').reset_index()
    new_df = new_df.rename_axis(None, axis=1)
    new_df.index.name = None
    new_df = new_df.apply(lambda x: x.fillna(-3))

    bureau_df = pd.read_csv('dataset/bureau.csv')[['SK_ID_CURR', 'SK_ID_BUREAU']]
    new_df = new_df.merge(bureau_df, on=['SK_ID_BUREAU'], how='left')

    application_train_df = pd.read_csv('dataset/application_train.csv')[['SK_ID_CURR', 'TARGET']]
    new_df = new_df.merge(application_train_df, on=['SK_ID_CURR'], how='left')

    new_df = new_df.dropna(subset=['SK_ID_CURR', 'TARGET'])
    new_df.to_csv('dataset/bureau_balance_processed.csv', index=False)
    print('Done.')
