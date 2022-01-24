import logging
from functools import partial

import jax
import jax.numpy as jnp
from jax import random

from abnercorrea.jax.optimizers.optimizer import Optimizer
from abnercorrea.jax.util.data_prep import create_sample_batch

logger = logging.getLogger(__name__)


class StochasticGradientDescent(Optimizer):
    """
    In both gradient descent (GD) and stochastic gradient descent (SGD), you update a set of parameters in an iterative manner to minimize an error function.

    While in GD, you have to run through ALL the samples in your training set to do a single update for a parameter in a particular iteration,
    in SGD, on the other hand, you use ONLY ONE or SUBSET of training sample from your training set to do the update for a parameter in a particular iteration.
    If you use SUBSET, it is called Minibatch Stochastic gradient Descent.

    https://datascience.stackexchange.com/a/36451
    https://web.archive.org/web/20180618211933/http://cs229.stanford.edu/notes/cs229-notes1.pdf
    """

    def __init__(self, mini_batch_size=256, epochs=10000, learning_rate=1e-2, tol=1e-3, overshoot_decrease_rate=.75, stagnation_batch_size=2, prng_key=None):
        assert mini_batch_size is None or mini_batch_size > 0, 'mini_batch_size can be either greater than 0 or None (for ordinary GD).'
        assert stagnation_batch_size > 1, 'stagnation_batch_size has to be greater than 1.'

        self.mini_batch_size = mini_batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.tol = tol
        self.overshoot_decrease_rate = overshoot_decrease_rate
        self.stagnation_batch_size = stagnation_batch_size
        self.prng_key = prng_key or random.PRNGKey(0)

    @staticmethod
    @partial(jax.jit, static_argnames=('loss_f',))
    def sgd_update(params, x, y, learning_rate, loss_f):
        loss, grads = jax.value_and_grad(loss_f)(params, x, y)

        params = jax.tree_multimap(lambda p, g: p - learning_rate * g, params, grads)

        return params, loss, grads

    def optimize(self, params, x, y, loss_f):
        mini_batch_size, epochs, tol, learning_rate, overshoot_decrease_rate, stagnation_batch_size = self.mini_batch_size, self.epochs, self.tol, self.learning_rate, self.overshoot_decrease_rate, self.stagnation_batch_size

        loss_hist = [loss_f(params, x, y)]

        for epoch in range(1, epochs + 1):
            # Creates mini batch
            x_batch, y_batch = create_sample_batch(x, y, mini_batch_size, self.prng_key)
            # Applies SGD update
            params, loss, grads = self.sgd_update(params, x_batch, y_batch, learning_rate, loss_f)
            # TODO: vanished gradient?
            # Vanishing gradients make it difficult to know which direction the parameters should move to improve the cost function … (from Deep Learning)
            if jnp.all(jnp.isnan(grads[0])).item():
                logger.error(f'Gradient vanished! - epoch: {epoch}, loss: {loss}, prev_loss: {loss_hist[-1]}')
                break
            loss_hist.append(loss)
            overshoot = loss_hist[-1] > loss_hist[-2]
            if overshoot:
                learning_rate *= overshoot_decrease_rate
                logger.info(f'Overshoot! Lowering learning rate to {learning_rate}')
            elif epoch >= stagnation_batch_size:
                loss_delta = jnp.abs(loss_hist[-1] - loss_hist[-stagnation_batch_size])
                if loss_delta < tol:
                    logger.info(f'SGD converged in {epoch} epochs!')
                    break

        return params


class GradientDescent(StochasticGradientDescent):
    def __init__(self, epochs=10000, learning_rate=1e-2, tol=1e-3, overshoot_decrease_rate=.75, stagnation_batch_size=2):
        super().__init__(
            mini_batch_size=None, 
            epochs=epochs, 
            learning_rate=learning_rate, 
            tol=tol, 
            overshoot_decrease_rate=overshoot_decrease_rate, 
            stagnation_batch_size=stagnation_batch_size
        )
