import pandas as pd

if __name__ == '__main__':
    bureau_balance_df = pd.read_csv('dataset/bureau_balance.csv')
    bureau_balance_df['MONTHS_BALANCE'] = bureau_balance_df['MONTHS_BALANCE'] * -1
    bureau_balance_df['STATUS_NUMBER'] = pd.to_numeric(
        bureau_balance_df['STATUS'].map({'C': 0,  # Pagamento feito normal
                                         'X': 0.1,  # Nao tem a informacao, vamos dar um risco baixo
                                         '0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5}))
    max_months = bureau_balance_df['MONTHS_BALANCE'].max()
    bureau_balance_df['RISK_SCORE'] = (
                (max_months - bureau_balance_df['MONTHS_BALANCE']) * bureau_balance_df['STATUS_NUMBER'])
    bureau_balance_df['RISK_SCORE'] = bureau_balance_df['RISK_SCORE'] / bureau_balance_df['RISK_SCORE'].max()

    new_df = bureau_balance_df.pivot(index='SK_ID_BUREAU', columns='MONTHS_BALANCE', values='STATUS_NUMBER').reset_index()
    new_df = new_df.rename_axis(None, axis=1)
    new_df.index.name = None
    new_df = new_df.apply(lambda x: x.fillna(-3))

    bureau_df = pd.read_csv('dataset/bureau.csv')[['SK_ID_CURR', 'SK_ID_BUREAU']]
    new_df = new_df.merge(bureau_df, on=['SK_ID_BUREAU'], how='left')
    new_df = new_df.merge(bureau_balance_df.groupby(by='SK_ID_BUREAU')['RISK_SCORE'].sum().reset_index(), on=['SK_ID_BUREAU'], how='left')

    application_train_df = pd.read_csv('dataset/application_train.csv')[['SK_ID_CURR', 'TARGET']]
    application_train_df['TRAIN'] = 1
    new_df = new_df.merge(application_train_df, on=['SK_ID_CURR'], how='left')

    application_test_df = pd.read_csv('dataset/application_test.csv')[['SK_ID_CURR']]
    new_df = new_df.merge(application_test_df, on=['SK_ID_CURR'], how='left')
    new_df['TRAIN'] = new_df['TRAIN'].fillna(0)

    #new_df = new_df.dropna(subset=['SK_ID_CURR', 'TARGET'])
    new_df.to_csv('dataset/bureau_balance_processed.csv', index=False)
    print('Done.')
