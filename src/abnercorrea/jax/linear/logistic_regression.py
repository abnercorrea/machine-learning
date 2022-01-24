import logging

import jax
import jax.numpy as jnp

from functools import partial

from abnercorrea.jax.optimizers.optimizer import Optimizer

logger = logging.getLogger(__name__)


class LogisticRegressionClassifierJax:
    def __init__(self, params, optimizer: Optimizer):
        self.params = params
        self.optimizer = optimizer

    @staticmethod
    @jax.jit
    def log_likelihood_loss(params, x, y):
        w, b = params
        p = jax.nn.sigmoid(jnp.dot(x, w) + b)
        # likelihood = jnp.product(p ** y * (1 - p) ** (1 - y))
        log_likelihood = jnp.sum(y * jnp.log(p) + (1 - y) * jnp.log(1 - p))
        return -log_likelihood

    @partial(jax.jit, static_argnames=('self',))
    def predict(self, x):
        w, b = self.params
        p = jax.nn.sigmoid(jnp.dot(x, w) + b)
        return jnp.where(p >= .5, 1, 0)

    def fit(self, x, y):
        self.params = self.optimizer.optimize(self.params, x, y, self.log_likelihood_loss)

    @partial(jax.jit, static_argnames=('self',))
    def accuracy(self, x, y):
        yp = self.predict(x)
        accurate = jnp.count_nonzero(yp == y)
        n = y.size
        return accurate / n
