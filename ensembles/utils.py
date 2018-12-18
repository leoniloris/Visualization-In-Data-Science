import pandas as pd
import numpy as np
import os
import itertools
import matplotlib.pyplot as plt

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

    return (x_train.values, y_train.values), \
        (x_val.values, y_val.values), \
        (x_test.values, y_test.values)


def split_train_val_test_sup_unsup(x, y, train_perc):
    (x_train_sup, y_train_sup), (x_valtest, y_valtest) = \
        _split_sets(x, y, train_perc)

    (x_val, y_val), (x_test, y_test) = \
        _split_sets(x_valtest, y_valtest, 0.5)

    (x_train_unsup, y_train_unsup) = (x_train_sup[y_train_sup.diagnosis == 'B'],
                                      y_train_sup[y_train_sup.diagnosis == 'B'])

    return (x_train_sup.values, y_train_sup.values), \
        (x_train_unsup.values, y_train_unsup.values), \
        (x_val.values, y_val.values), \
        (x_test.values, y_test.values)


def plot_confusion_matrix(cm, classes):
    '''
    usage: plot_confusion_matrix(confusion_matrix(outliers_idx, real_outliers_idx), [False, True])
    '''
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], '.2f'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
