# Geracao de features baseadas no bureau.csv e bureau_balance.csv
# Baseado em: https://www.kaggle.com/shanth84/credit-card-balance-feature-engineering/

import pandas as pd


def is_active(x):
    if x == 'Closed':
        return 0
    return 1


def is_positive(x):
    if x < 0:
        return 0
    return 1


def bureau_feature_statistics(input_df, output_df, feature_name, agg_functions=None):
    columns = {}
    if agg_functions is None:
        agg_functions = ['min']
        columns['min'] = feature_name
    else:
        for f in agg_functions:
            columns[f] = f.upper() + '_' + feature_name
    group = input_df.groupby(by=['SK_ID_CURR'])[feature_name].agg(agg_functions).reset_index().rename(index=str,
                                                                                                      columns=columns)
    if output_df is not None:
        return output_df.merge(group, on=['SK_ID_CURR'], how='left')
    return group


def prepare_bureau_data():
    bureau_df = pd.read_csv('dataset/bureau.csv')
    new_df = bureau_df[['SK_ID_CURR', 'SK_ID_BUREAU']]

    print('Number of past loans per customer')
    grp = bureau_df[['SK_ID_CURR', 'DAYS_CREDIT']].groupby(by=['SK_ID_CURR'])[
        'DAYS_CREDIT'].count().reset_index().rename(index=str, columns={'DAYS_CREDIT': 'BUREAU_LOAN_COUNT'})
    new_df = new_df.merge(grp, on=['SK_ID_CURR'], how='left')

    print('Number of Types of Past Loans per Customer')
    grp = bureau_df[['SK_ID_CURR', 'CREDIT_TYPE']].groupby(by=['SK_ID_CURR'])[
        'CREDIT_TYPE'].nunique().reset_index().rename(index=str, columns={'CREDIT_TYPE': 'BUREAU_LOAN_TYPES'})
    new_df = new_df.merge(grp, on=['SK_ID_CURR'], how='left')

    print('Average Number of Loans per Loan Type')
    new_df['AVERAGE_LOAN_TYPE'] = new_df['BUREAU_LOAN_COUNT'] / new_df['BUREAU_LOAN_TYPES']

    print('Calculate mean number of loans that are ACTIVE per CUSTOMER')
    bureau_df['CREDIT_ACTIVE_BINARY'] = bureau_df.apply(lambda x: is_active(x.CREDIT_ACTIVE), axis=1)
    grp = bureau_df.groupby(by=['SK_ID_CURR'])['CREDIT_ACTIVE_BINARY'].mean().reset_index().rename(index=str, columns={
        'CREDIT_ACTIVE_BINARY': 'ACTIVE_LOANS_PERCENTAGE'})
    new_df = new_df.merge(grp, on=['SK_ID_CURR'], how='left')
    del bureau_df['CREDIT_ACTIVE_BINARY']

    # How often did the customer take credit in the past? Was it spaced out at regular time intervals - a signal of good
    # ... financial planning OR were the loans concentrated around a smaller time frame - indicating potential financial
    # ... trouble?
    print('Check customer take credit frequency')
    grp = bureau_df[['SK_ID_CURR', 'SK_ID_BUREAU', 'DAYS_CREDIT']].groupby(by=['SK_ID_CURR'])
    grp1 = grp.apply(lambda x: x.sort_values(['DAYS_CREDIT'], ascending=False)).reset_index(drop=True)
    grp1['DAYS_CREDIT1'] = grp1['DAYS_CREDIT'] * -1
    grp1['DAYS_DIFF'] = grp1.groupby(by=['SK_ID_CURR'])['DAYS_CREDIT1'].diff()
    grp1['DAYS_DIFF'] = grp1['DAYS_DIFF'].fillna(0).astype('uint32')
    del grp1['DAYS_CREDIT1'], grp1['DAYS_CREDIT'], grp1['SK_ID_CURR']
    new_df = new_df.merge(grp1, on=['SK_ID_BUREAU'], how='left')

    # % of LOANS PER CUSTOMER WHERE END DATE FOR CREDIT IS PAST
    # NEGATIVE VALUE - Credit date was in the past at time of application( Potential Red Flag !!! )
    # POSITIVE VALUE - Credit date is in the future at time of application ( Potential Good Sign !!!!)
    # NOTE : This is not the same as % of Active loans since Active loans
    # can have Negative and Positive values for DAYS_CREDIT_ENDDATE
    print('% of loans per customer where end date for credit is past')
    bureau_df['CREDIT_ENDDATE_BINARY'] = bureau_df['DAYS_CREDIT_ENDDATE']
    bureau_df['CREDIT_ENDDATE_BINARY'] = bureau_df.apply(lambda x: is_positive(x.DAYS_CREDIT_ENDDATE), axis=1)
    grp = bureau_df.groupby(by=['SK_ID_CURR'])['CREDIT_ENDDATE_BINARY'].mean().reset_index().rename(index=str, columns={
        'CREDIT_ENDDATE_BINARY': 'CREDIT_ENDDATE_PERCENTAGE'})
    new_df = new_df.merge(grp, on=['SK_ID_CURR'], how='left')

    # Average Number of Days in Which Credit Expires in Future - Indication of Customer Delinquency in Future?
    print('Average Number of Days in Which Credit Expires in Future - Indication of Customer Delinquency in Future?')
    B1 = bureau_df[bureau_df['CREDIT_ENDDATE_BINARY'] == 1]
    B1['DAYS_CREDIT_ENDDATE1'] = B1['DAYS_CREDIT_ENDDATE']
    grp = B1[['SK_ID_CURR', 'SK_ID_BUREAU', 'DAYS_CREDIT_ENDDATE1']].groupby(by=['SK_ID_CURR'])
    grp1 = grp.apply(lambda x: x.sort_values(['DAYS_CREDIT_ENDDATE1'], ascending=True)).reset_index(drop=True)
    del grp
    grp1['DAYS_ENDDATE_DIFF'] = grp1.groupby(by=['SK_ID_CURR'])['DAYS_CREDIT_ENDDATE1'].diff()
    grp1['DAYS_ENDDATE_DIFF'] = grp1['DAYS_ENDDATE_DIFF'].fillna(0).astype('uint32')
    del grp1['DAYS_CREDIT_ENDDATE1'], grp1['SK_ID_CURR']
    new_df = new_df.merge(grp1, on=['SK_ID_BUREAU'], how='left')
    grp = new_df[['SK_ID_CURR', 'DAYS_ENDDATE_DIFF']].groupby(by=['SK_ID_CURR'])[
        'DAYS_ENDDATE_DIFF'].mean().reset_index().rename(index=str, columns={'DAYS_ENDDATE_DIFF': 'AVG_ENDDATE_FUTURE'})
    new_df = new_df.merge(grp, on=['SK_ID_CURR'], how='left')

    # The Ratio of Total Debt to Total Credit for each Customer
    # A High value may be a red flag indicative of potential default
    print('Debt over credit ratio')
    bureau_df['AMT_CREDIT_SUM_DEBT'] = bureau_df['AMT_CREDIT_SUM_DEBT'].fillna(0)
    bureau_df['AMT_CREDIT_SUM'] = bureau_df['AMT_CREDIT_SUM'].fillna(0)

    grp1 = bureau_df[['SK_ID_CURR', 'AMT_CREDIT_SUM_DEBT']].groupby(by=['SK_ID_CURR'])[
        'AMT_CREDIT_SUM_DEBT'].sum().reset_index().rename(index=str,
                                                          columns={'AMT_CREDIT_SUM_DEBT': 'TOTAL_CUSTOMER_DEBT'})
    grp2 = bureau_df[['SK_ID_CURR', 'AMT_CREDIT_SUM']].groupby(by=['SK_ID_CURR'])[
        'AMT_CREDIT_SUM'].sum().reset_index().rename(index=str, columns={'AMT_CREDIT_SUM': 'TOTAL_CUSTOMER_CREDIT'})
    new_df = new_df.merge(grp1, on=['SK_ID_CURR'], how='left')
    new_df = new_df.merge(grp2, on=['SK_ID_CURR'], how='left')
    del grp1, grp2

    new_df['DEBT_CREDIT_RATIO'] = new_df['TOTAL_CUSTOMER_DEBT'] / new_df['TOTAL_CUSTOMER_CREDIT']

    # OVERDUE OVER DEBT RATIO
    # What fraction of total Debt is overdue per customer?
    print('Overdue over debt ratio')
    grp2 = bureau_df[['SK_ID_CURR', 'AMT_CREDIT_SUM_OVERDUE']].groupby(by=['SK_ID_CURR'])[
        'AMT_CREDIT_SUM_OVERDUE'].sum().reset_index().rename(index=str, columns={
        'AMT_CREDIT_SUM_OVERDUE': 'TOTAL_CUSTOMER_OVERDUE'})
    new_df = new_df.merge(grp2, on=['SK_ID_CURR'], how='left')
    new_df['OVERDUE_DEBT_RATIO'] = new_df['TOTAL_CUSTOMER_OVERDUE'] / new_df['TOTAL_CUSTOMER_DEBT']

    # AVERAGE NUMBER OF LOANS PROLONGED
    print('Average number of loans prolonged')
    bureau_df['CNT_CREDIT_PROLONG'] = bureau_df['CNT_CREDIT_PROLONG'].fillna(0)
    grp = bureau_df[['SK_ID_CURR', 'CNT_CREDIT_PROLONG']].groupby(by=['SK_ID_CURR'])[
        'CNT_CREDIT_PROLONG'].mean().reset_index().rename(index=str,
                                                          columns={'CNT_CREDIT_PROLONG': 'AVG_CREDITDAYS_PROLONGED'})
    new_df = new_df.merge(grp, on=['SK_ID_CURR'], how='left')

    print('Collecting credit risk score')
    bureau_balance_df = pd.read_csv('dataset/processed_bureau_balance.csv')
    new_df = new_df.merge(bureau_balance_df[['SK_ID_BUREAU', 'RISK_SCORE']], on=['SK_ID_BUREAU'], how='left')

    # Gera features finais com informacoes estatisticas
    final_df = bureau_feature_statistics(new_df, None, 'BUREAU_LOAN_COUNT')
    final_df = bureau_feature_statistics(new_df, final_df, 'BUREAU_LOAN_TYPES')
    final_df = bureau_feature_statistics(new_df, final_df, 'AVERAGE_LOAN_TYPE')
    final_df = bureau_feature_statistics(new_df, final_df, 'ACTIVE_LOANS_PERCENTAGE')
    final_df = bureau_feature_statistics(new_df, final_df, 'DAYS_DIFF', ['min', 'max', 'mean', 'std'])
    final_df = bureau_feature_statistics(new_df, final_df, 'CREDIT_ENDDATE_PERCENTAGE')
    final_df = bureau_feature_statistics(new_df, final_df, 'DAYS_ENDDATE_DIFF', ['min', 'max', 'mean', 'std'])
    final_df = bureau_feature_statistics(new_df, final_df, 'AVG_ENDDATE_FUTURE')
    final_df = bureau_feature_statistics(new_df, final_df, 'TOTAL_CUSTOMER_DEBT')
    final_df = bureau_feature_statistics(new_df, final_df, 'TOTAL_CUSTOMER_CREDIT')
    final_df = bureau_feature_statistics(new_df, final_df, 'DEBT_CREDIT_RATIO')
    final_df = bureau_feature_statistics(new_df, final_df, 'TOTAL_CUSTOMER_OVERDUE')
    final_df = bureau_feature_statistics(new_df, final_df, 'OVERDUE_DEBT_RATIO')
    final_df = bureau_feature_statistics(new_df, final_df, 'AVG_CREDITDAYS_PROLONGED')
    final_df = bureau_feature_statistics(new_df, final_df, 'RISK_SCORE', ['min', 'max', 'mean', 'std', 'sum'])
    final_df.to_csv('dataset/processed_bureau.csv', index=False)

    print('Done.')
