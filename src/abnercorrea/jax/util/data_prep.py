import numpy as np
from jax import random


def create_sample_batch(x, y, mini_batch_size, prng_key):
    if mini_batch_size is None:
        return x, y

    n = y.size
    sample_indices = random.choice(prng_key, n, shape=(mini_batch_size,), replace=False).tolist()
    x_batch = x[sample_indices]
    y_batch = y[sample_indices]

    return x_batch, y_batch


def init_nn_params(layer_widths):
    params = []
    for n_in, n_out in zip(layer_widths[:-1], layer_widths[1:]):
        params.append(
            dict(
                weights=np.random.normal(size=(n_in, n_out)) * np.sqrt(2 / n_in),
                biases=np.ones(shape=(n_out,))
            )
        )
    return params
