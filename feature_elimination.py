# Remover features que possuem alta correlação entre si (0.7)
import pandas as pd
import data


def identify_features_to_remove(correlation_df, train_df, min_corr, verbose=False):
    features_list = []
    ignore_list = []
    for column_name in train_df.columns:
        first = True
        for index in correlation_df[correlation_df[column_name] >= min_corr].index:
            if train_df.columns[index] in ignore_list:
                continue
            if train_df.columns[index] == column_name:
                continue

            if verbose:
                if first:
                    print('Column %s' % column_name)
                    first = False
                print('\t', correlation_df.columns[index],
                      train_df[column_name].corr(train_df[train_df.columns[index]]))
            ignore_list.append(column_name)
            features_list.append(column_name)
    return list(set(features_list))


if __name__ == '__main__':
    train_df, test_df = data.load_dataset()
    corr = pd.read_csv('dataset/correlations.csv')

    features_to_remove = identify_features_to_remove(corr, train_df, 1, verbose=True)
    print(len(train_df.columns), len(features_to_remove), features_to_remove)
