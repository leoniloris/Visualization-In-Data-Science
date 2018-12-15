import pandas as pd
import numpy as np
import os


def _preprocess_data(df):
    df /= df.max(axis=0)
    return df


def load_database():
    os.system(
        'kaggle datasets download '
        '-d uciml/breast-cancer-wisconsin-data -f data.csv')
    return pd.read_csv('data.csv')


def create_xy(df, add_columns, target_columns):
    columns = [
        col for col in df.columns if
        ((col not in target_columns)) and
        ((add_columns in col) or (add_columns in col))
    ]

    return df[columns], df[target_columns]


def split_train_val(x_df, y_df):
    _x_df = _preprocess_data(x_df.copy())
    TRAIN_SIZE = int(0.8 * len(_x_df))
    train_idxs = np.random.choice(_x_df.index, TRAIN_SIZE, replace=False)
    val_idxs = _x_df.index.drop(train_idxs)
    x_train = x_df.loc[train_idxs].values
    y_train = y_df.loc[train_idxs].values
    print('# Examples for training:', len(x_train))

    x_val = _x_df.loc[val_idxs].values
    y_val = y_df.loc[val_idxs].values
    print('# Examples for validation:', len(x_val))
    return (x_train, y_train), (x_val, y_val)
