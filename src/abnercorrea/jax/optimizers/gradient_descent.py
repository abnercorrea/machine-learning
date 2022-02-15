from absl import logging
from functools import partial

import jax
import jax.numpy as jnp
from jax import random

from abnercorrea.jax.optimizers.optimizer import Optimizer
from abnercorrea.jax.util.data_prep import create_sample_batch


class MiniBatchSGD(Optimizer):
    """
    In both gradient descent (GD) and stochastic gradient descent (SGD), you update a set of parameters in an iterative manner to minimize an error function.

    While in GD, you have to run through ALL the samples in your training set to do a single update for a parameter in a particular iteration,
    in SGD, on the other hand, you use ONLY ONE or SUBSET of training sample from your training set to do the update for a parameter in a particular iteration.
    If you use SUBSET, it is called Minibatch Stochastic gradient Descent.

    IMPORTANT: it's common practice to choose the mini batch size to be a power of 2. (usually between 64 and 512)

    https://datascience.stackexchange.com/a/36451
    https://web.archive.org/web/20180618211933/http://cs229.stanford.edu/notes/cs229-notes1.pdf

    # TODO: implement learning rate decay:
        - lr = lr * (1 / 1 + decay_rate * epoch)
        - lr = lr * 0.95 ** epoch  (exponential decay)
        - lr = lr * (k / epoch ** .5)
        - discrete staircase.
    """

    def __init__(self, mini_batch_size=256, epochs=10000, learning_rate=1e-8, eps=1e-3, overshoot_decrease_rate=.5, stagnation_batch_size=2, prng_key=None):
        assert mini_batch_size is None or mini_batch_size > 0, 'mini_batch_size can be either greater than 0 or None (for ordinary GD).'
        assert stagnation_batch_size > 1, 'stagnation_batch_size has to be greater than 1.'

        self.mini_batch_size = mini_batch_size
        self.epochs = epochs
        self.epoch = 0
        self.learning_rate = learning_rate
        self.eps = eps
        self.overshoot_decrease_rate = overshoot_decrease_rate
        self.stagnation_batch_size = stagnation_batch_size
        self.prng_key = prng_key or random.PRNGKey(0)

    @partial(jax.jit, static_argnames=('self',))
    def calculate_update(self, grads, epoch):
        # Ordinary SGD uses gradient for updates
        return grads

    @partial(jax.jit, static_argnames=('self', 'loss_f',))
    def sgd_update(self, params, x, y, loss_f):
        loss, grads = jax.value_and_grad(loss_f)(params, x, y)

        logging.debug(f'Epoch: {self.epoch}\nLoss: {loss}\nGrads: {grads}\nParams: {params}')

        update = self.calculate_update(grads, self.epoch)

        params = jax.tree_multimap(lambda p, g: p - self.learning_rate * g, params, update)

        logging.debug(f'Params updated: {params}')

        return params, loss, grads

    def optimize(self, params, x, y, loss_f):
        """
        Vanishing gradients make it difficult to know which direction the parameters should move to improve the cost function … (from Deep Learning)

        :param params:
        :param x:
        :param y:
        :param loss_f:
        :return:
        """
        mini_batch_size, epochs, eps, learning_rate, overshoot_decrease_rate, stagnation_batch_size = self.mini_batch_size, self.epochs, self.eps, self.learning_rate, self.overshoot_decrease_rate, self.stagnation_batch_size

        loss_hist = []

        for epoch in range(1, epochs + 1):
            self.epoch = epoch

            # Creates mini batch
            x_batch, y_batch = create_sample_batch(x, y, mini_batch_size, self.prng_key)
            # Applies SGD update
            params, loss, grads = self.sgd_update(params, x_batch, y_batch, loss_f)
            loss_hist.append(loss.item())

            logging.debug(f'Loss hist: {loss_hist}')

            # TODO: vanished gradient? diverged?
            if jnp.all(jnp.isnan(grads[0])).item():
                logging.error(f'Gradient vanished! - epoch: {epoch}, loss: {loss}, prev_loss: {loss_hist[-2] if epoch > 1 else None}')
                break

            # TODO: research and improve learning rate decreasing
            if epoch > 1:
                overshoot = loss_hist[-1] > loss_hist[-2]
                if overshoot:
                    learning_rate *= overshoot_decrease_rate
                    logging.info(f'Overshoot! Lowering learning rate to {learning_rate} - Epoch: {epoch} - Loss: {loss} - Prev loss: {loss_hist[-2]}')

            # TODO: research and improve convergence checking
            if epoch >= stagnation_batch_size:
                loss_delta = jnp.abs(loss_hist[-1] - loss_hist[-stagnation_batch_size])
                logging.debug(f'Loss delta: {loss_delta}')
                if loss_delta <= eps:
                    logging.info(f'SGD converged in {epoch} epochs! - Initial loss: {loss_hist[0]}, final loss: {loss}')
                    break

        return params


class GradientDescent(MiniBatchSGD):
    def __init__(self, **kwargs):
        """
        In ordinary gradient descent, the batch size is equal to the entire training set.
        """
        assert 'mini_batch_size' not in kwargs
        super().__init__(mini_batch_size=None, **kwargs)


class SGD(MiniBatchSGD):
    def __init__(self, **kwargs):
        """
        In stochastic gradient descent, the batch size is equal to 1.
        """
        assert 'mini_batch_size' not in kwargs
        super().__init__(mini_batch_size=1, **kwargs)
