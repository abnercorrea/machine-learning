import logging

import jax
import jax.numpy as jnp

from abnercorrea.jax.optimizers.optimizer import Optimizer
from abnercorrea.jax.util.data_prep import init_nn_params

logger = logging.getLogger(__name__)


class MLP:
    default_layer_widths = [1, 128, 128, 1]

    def __init__(self, params, optimizer: Optimizer, layer_widths=None):
        self.params = params or init_nn_params(self.layer_widths)
        self.optimizer = optimizer
        self.layer_widths = layer_widths or self.default_layer_widths

    @staticmethod
    @jax.jit
    def forward(params, x):
        *hidden, last = params
        for layer in hidden:
            x = jax.nn.relu(x @ layer['weights'] + layer['biases'])
        return x @ last['weights'] + last['biases']

    @staticmethod
    @jax.jit
    def mean_squared_error(params, x, y):
        mse = jnp.mean((MLP.forward(params, x) - y) ** 2)
        return mse

    def fit(self, x, y):
        self.params = self.optimizer.optimize(self.params, x, y, self.mean_squared_error)

    def predict(self, x):
        return self.forward(self.params, x)
