import numpy as np


def weighted_mode(a, w, axis=0):
    """
    Based on the beatiful sklearn.utils.extmath.weighted_mode... :)
    """
    assert a.shape == w.shape, 'Array of labels and array of weights must have the same shape.'

    unique_values = np.unique(np.ravel(a))

    # This operation reduces the values on the given axis to the value with largest weight sum (thus 1)
    result_shape = list(a.shape)
    result_shape[axis] = 1
    # Used to store values with max weight sum
    max_values = np.zeros(result_shape)
    # Used to store the max weight sums
    max_w_sum = np.zeros(result_shape, dtype=np.float64)

    for value in unique_values:
        # creates a mask (array of booleans) used to filter weights (w) of matching elements of a that are == value
        value_mask = (a == value)
        # creates a template with same shape of a to receive filtered elements by value
        w_filtered = np.zeros(a.shape, dtype=np.float64)
        # filter only weights where corresponding element in a == value
        w_filtered[value_mask] = w[value_mask]
        # sums weights (used to determine max)
        w_sum = np.expand_dims(np.sum(w_filtered, axis, dtype=np.float64), axis)
        # Updates max weight sums and matching values using a mask.
        max_mask = (w_sum > max_w_sum)
        max_values[max_mask] = value
        max_w_sum[max_mask] = w_sum[max_mask]

    return max_values, max_w_sum
