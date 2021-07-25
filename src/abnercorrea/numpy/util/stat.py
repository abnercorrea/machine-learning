import numpy as np


def logistic_sigmoid(x):
    return 1. / (1. + np.exp(-x))


def softmax(x):
    # subtracts max value to prevent potential overflow in case of large values of x.
    numerator = np.exp(x - np.max(x, axis=-1)[:, np.newaxis])
    denominator = numerator.sum(axis=-1)[:, np.newaxis]
    return numerator / denominator


def empirical_covariance(X, y, classes, mu):
    x = np.copy(X)
    for class_k, mu_k in zip(classes, mu):
        x[y == class_k] -= mu_k
    return (x.T @ x) / x.shape[0]


def mean_squared_error(y, y_pred):
    mse = np.average((y - y_pred) ** 2)
    return mse


def r2_score(y, y_pred):
    """
    "R squared" provides a measure of how well observed outcomes are replicated by the model,
    based on the proportion of total variation of outcomes explained by the model.
    """
    numerator = ((y - y_pred) ** 2).sum(axis=0)
    denominator = ((y - np.average(y, axis=0)) ** 2).sum(axis=0)
    if numerator == 0:
        return 1
    if denominator == 0:
        return 0
    return 1 - numerator / denominator
