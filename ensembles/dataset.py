import pandas as pd
import numpy as np
import os

np.random.seed(0)

def preprocess_data(df):
    df /= df.max(axis=0)
    return df


def load_dataset():
    os.system(
        'kaggle datasets download '
        '-d uciml/breast-cancer-wisconsin-data -f data.csv')
    return pd.read_csv('data.csv')


def create_xy(df, add_columns, target_columns):
    columns = [
        col for col in df.columns if
        (col not in target_columns and 
         any(add in col for add in add_columns))
    ]

    return df[columns], df[target_columns]


def _split_sets(x, y, first_set_perc):
    first_set_size = int(first_set_perc * len(x))
    
    first_set_idxs = np.random.choice(x.index, first_set_size, replace=False)
    second_set_idxs = x.index.drop(first_set_idxs)
    
    x_first_set = x.loc[first_set_idxs]
    y_first_set = y.loc[first_set_idxs]
    
    x_second_set = x.loc[second_set_idxs]
    y_second_set = y.loc[second_set_idxs]
    
    return (x_first_set, y_first_set), (x_second_set, y_second_set)
    
    
def split_train_val_test(x, y, train_perc):
    x_healthy, y_healthy = x[y.diagnosis == 'B'], y[y.diagnosis == 'B']
    x_unhealthy, y_unhealthy = x[y.diagnosis == 'M'], y[y.diagnosis == 'M']
    
    (x_train, y_train), (x_valtest, y_valtest) = \
        _split_sets(x_healthy, y_healthy, train_perc)

    x_valtest = pd.concat([x_valtest, x_unhealthy], ignore_index=True)
    y_valtest = pd.concat([y_valtest, y_unhealthy], ignore_index=True)

    (x_val, y_val), (x_test, y_test) = \
        _split_sets(x_valtest, y_valtest, 0.5)
    
    return (x_train.values, y_train.values), (x_val.values, y_val.values), (x_test.values, y_test.values)
