from functools import partial

import jax
import jax.numpy as jnp

from abnercorrea.jax.optimizers.gradient_descent import MiniBatchSGD


class RMSProp(MiniBatchSGD):
    """
    Uses the exponentially weighted average of the square of the gradients to update parameters.
    """
    def __init__(self, momentum=0.9, rms_eps=1e-8, **kwargs):
        super().__init__(**kwargs)

        self.momentum = momentum
        self.rms_eps = rms_eps
        # Exponentially weighted average of the square of the gradients
        self.sd = None


    @partial(jax.jit, static_argnames=('self',))
    def calculate_update(self, grads, epoch):
        sd = self.sd or jax.tree_multimap(lambda g: jnp.zeros_like(g), grads)

        # exponentially weighted average of the square of the gradients
        sd = jax.tree_multimap(lambda sd, g: self.momentum * sd + (1 - self.momentum) * g ** 2, sd, grads)

        # RMSProp update
        update = jax.tree_multimap(lambda sd, g: g / (sd ** 0.5 + self.rms_eps), sd, grads)

        self.sd = sd

        return update
