
import pandas as pd

def pandas_multi_column_agg(df, columns_to_agg):
    rc_results = pd.DataFrame()
    other_columns = list(set(df.columns) - set(columns_to_agg))
    rc_results['support'] = df.groupby(columns_to_agg).count()[other_columns[0]]
    rc_results['precentage%'] = rc_results['support'].groupby(level=0).apply(lambda x:
                                                                             100 * x / float(x.sum())).astype(int)
    return rc_results

def uniform_sample_by_column(data, weights_column, key_column, n, random_state=17):
    data_weighted = data.copy(deep=True).drop_duplicates(subset=[key_column])

    weights = 1 / data[weights_column].value_counts()
    weights = weights / sum(weights)
    data_weighted = data_weighted.set_index(weights_column)
    data_weighted.loc[weights.index, 'weights'] = weights
    data_weighted = data_weighted.reset_index()

    if len(data_weighted) > n:
        sample = data_weighted.sample(n=n, weights='weights', random_state=random_state)
    else:
        sample = data

    data = data[data[key_column].isin(sample[key_column])]
    # logger.info("uniform sampling by %s #sample from each value: \n%s" % (weights_column, sample[weights_column].value_counts()))
    return data