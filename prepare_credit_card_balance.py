import gc
import pandas as pd
from cat_features import one_hot_encoder


def prepare_credit_card_balance(num_rows=None, nan_as_category=True):
    cc = pd.read_csv('dataset/credit_card_balance.csv', nrows=num_rows)
    cc, cat_cols = one_hot_encoder(cc, nan_as_category=True)
    # General aggregations
    cc.drop(['SK_ID_PREV'], axis=1, inplace=True)
    cc_agg = cc.groupby('SK_ID_CURR').agg(['min', 'max', 'mean', 'sum', 'var'])
    cc_agg.columns = pd.Index(['CC_' + e[0] + "_" + e[1].upper() for e in cc_agg.columns.tolist()])
    # Count credit card lines
    cc_agg['CC_COUNT'] = cc.groupby('SK_ID_CURR').size()
    del cc
    gc.collect()
    cc_agg.to_csv('dataset/processed_credit_card_balance.csv', index=False)


if __name__ == '__main__':
    prepare_credit_card_balance()
