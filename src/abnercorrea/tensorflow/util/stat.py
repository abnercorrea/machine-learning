import tensorflow as tf
import tensorflow.experimental.numpy as tnp
from tensorflow.python.ops.numpy_ops import np_config

np_config.enable_numpy_behavior()


def logistic_sigmoid(x):
    return 1. / (1. + tf.exp(-x))


def softmax(x):
    # subtracts max value to prevent potential overflow in case of large values of x.
    numerator = tf.exp(x - tf.reduce_max(x, axis=-1)[:, tf.newaxis])
    denominator = tf.reduce_sum(numerator, axis=-1)[:, tf.newaxis]
    return numerator / denominator


def empirical_covariance(X, y, classes, mu):
    x = tnp.copy(X)
    for class_k, mu_k in zip(classes, mu):
        x[y == class_k] -= mu_k
    return (x.T @ x) / x.shape[0]


def mean_squared_error(y, y_pred):
    mse = tnp.average((y - y_pred) ** 2)
    return mse


def r2_score(y, y_pred):
    """
    "R squared" provides a measure of how well observed outcomes are replicated by the model,
    based on the proportion of total variation of outcomes explained by the model.
    """
    numerator = tnp.sum((y - y_pred) ** 2)
    denominator = tnp.sum((y - tnp.average(y)) ** 2)
    score = tf.where(
        numerator == 0,
        tnp.float64(1),
        tf.where(
            denominator == 0,
            tnp.float64(0),
            1. - numerator / denominator
        )
    )
    return score
