import tensorflow as tf
import tensorflow.experimental.numpy as tnp
from tensorflow.python.ops.numpy_ops import np_config

np_config.enable_numpy_behavior()


@tf.function(
    input_signature=[
                    tf.TensorSpec(shape=None, dtype=tf.float64),
                    tf.TensorSpec(shape=None, dtype=tf.float64),
                    tf.TensorSpec(shape=(), dtype=tf.int32),
                    ]
            )
def weighted_mode(a, w, axis):
    """
    Based on the beautiful sklearn.utils.extmath.weighted_mode... :)
    """
    assert a.shape == w.shape, 'Array of labels and array of weights must have the same shape.'
    unique_values, _ = tf.unique(tnp.ravel(a))

    # This operation reduces the values on the given axis to the value with largest weight sum (thus 1)
    result_shape = (
        tf
        .TensorArray(tf.int32, size=tf.size(a))
        .unstack(tf.shape(a))
        .write(axis, 1)
        .stack()
    )
    # Used to store values with max weight sum
    max_values = tf.zeros(result_shape, dtype=tf.float64)
    # Used to store the max weight sums
    max_w_sum = tf.zeros(result_shape, dtype=tf.float64)

    for value in unique_values:
        # filter only weights where corresponding element in a == value
        w_filtered = tf.where(a == value, w, 0)
        # sums weights (used to determine max)
        w_sum = tnp.expand_dims(tnp.sum(w_filtered, axis), axis)
        # Updates max weight sums and matching values using a mask.
        max_mask = (w_sum > max_w_sum)
        max_values = tf.where(max_mask, value, max_values)
        max_w_sum = tf.where(max_mask, w_sum, max_w_sum)

    return max_values, max_w_sum
