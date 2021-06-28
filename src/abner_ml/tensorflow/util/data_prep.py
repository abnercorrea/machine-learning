import tensorflow as tf
import tensorflow.experimental.numpy as tnp
from tensorflow.python.ops.numpy_ops import np_config

np_config.enable_numpy_behavior()


def prepend_col(matrix, value):
    # padding = [[p11, p12], [p21, p22]]
    # p11 - padding before the first dimension
    # p12 - padding after the first dimension
    # p21 - padding before the second dimension
    # p22 - padding after the second dimension
    padding = [[0,0],[1,0]]
    return tf.pad(matrix, padding, constant_values=value)


@tf.function(
    input_signature=[
                    tf.TensorSpec(shape=None, dtype=tf.float64),
                    tf.TensorSpec(shape=None, dtype=tf.float64),
                    tf.TensorSpec(shape=(), dtype=tf.int32),
                    tf.TensorSpec(shape=(), dtype=tf.int32),
                    ]
            )
def split_train_validation_tf(x, y, fold, folds):
    """
    Splits input data into train and validation sets.
    Used in k-fold cross validation.
    """
    print("tracing split_train_validation_tf.")
    n = tf.shape(x)[0]
    fold_size = n // folds
    fold_start = fold * fold_size
    fold_end = (fold + 1) * fold_size
    train_x = tf.concat([x[:fold_start], x[fold_end:]], axis=0)
    train_y = tf.concat([y[:fold_start], y[fold_end:]], axis=0)
    validation_x, validation_y = x[fold_start:fold_end], y[fold_start:fold_end]
    return train_x, train_y, validation_x, validation_y


def norm(v):
    norm_v = tf.norm(v, axis=1)[:, tnp.newaxis]
    norm_v = tf.where(norm_v == 0, 1, norm_v)
    return v / norm_v
