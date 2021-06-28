import numpy as np
import pandas as pd


def read_csv(file_name_prefix, partition=None):
    # np.genfromtxt('trainData1.csv', dtype=int, delimiter=',')
    return pd.read_csv(f'{file_name_prefix}{partition or ""}.csv', header=None).values


def read_train_data(num_partitions):
    train_data = [read_csv('trainData', i + 1) for i in range(num_partitions)]
    train_labels = [read_csv('trainLabels', i + 1) for i in range(num_partitions)]
    # y is a scalar
    return train_data, train_labels


def read_test_data():
    test_data = read_csv('testData')
    test_labels = read_csv('testLabels')
    # y is a scalar
    return test_data, test_labels


def norm(v):
    norm_v = np.linalg.norm(v, axis=1)[:, np.newaxis]
    norm_v[norm_v == 0] = 1
    return v / norm_v


def prepend_row(matrix, value):
    r, c = matrix.shape
    new_matrix = np.zeros([r + 1, c])
    new_matrix[0, :] = value
    new_matrix[1:, :] = matrix
    return new_matrix


def prepend_col(matrix, value):
    r, c = matrix.shape
    new_matrix = np.zeros([r, c + 1])
    new_matrix[:, 0] = value
    new_matrix[:, 1:] = matrix
    return new_matrix


def to_binary_classes(y):
    """
    Used to build labels vector (y) in binary classification models.
    """
    classes = np.unique(y.ravel())
    assert classes.size == 2, f'Should contain only 2 classes but found {classes.size}: {classes}.'
    yb = np.zeros_like(y, dtype=np.int8)
    yb[y == classes[0]] = 1
    return yb, classes


def split_train_validation(x, y, fold, folds):
    """
    Splits input data into train and validation sets.
    Used in k-fold cross validation.

    For cross validation, when spliting train x validation, consider using stratified sampling:
    https://danilzherebtsov.medium.com/continuous-data-stratification-c121fc91964b
    """
    n = x.shape[0]
    fold_size = n // folds
    fold_start = fold * fold_size
    fold_end = (fold + 1) * fold_size
    xtr = np.concatenate([x[:fold_start], x[fold_end:]])
    ytr = np.concatenate([y[:fold_start], y[fold_end:]])
    xvl, yvl = x[fold_start:fold_end], y[fold_start:fold_end]
    return xtr, ytr, xvl, yvl


def center(v):
    """
    Centers variable. (subtracts the mean)
    The mean of the variable becomes 0 (centered)
    """
    return v - v.mean(axis=0)


def standardize(v):
    """
    Subtracts mean and divides by standard deviation. (centers and scales)
    - The convention that you standardize predictions primarily exists so that the units of the regression coefficients are the same.
    - Centering/scaling does not affect your statistical inference in regression models.
    - The estimates are adjusted appropriately and the 𝑝-values will be the same.
    https://stats.stackexchange.com/a/29783
    """
    return center(v) / v.std(axis=0)


def scale(v, factor):
    """
    This is not really a useful method... just to illustrate what scaling is...
    """
    return v * factor

