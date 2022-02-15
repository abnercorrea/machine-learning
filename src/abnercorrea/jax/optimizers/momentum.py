from functools import partial

import jax
import jax.numpy as jnp

from abnercorrea.jax.optimizers.gradient_descent import MiniBatchSGD


class SGDWithMomentum(MiniBatchSGD):
    """
    Uses an exponentially weighted average of the gradients to update parameters.
    """
    def __init__(self, momentum=0.9, **kwargs):
        super().__init__(**kwargs)

        self.momentum = momentum
        # Exponentially weighted average of the gradients
        self.vd = None

    @partial(jax.jit, static_argnames=('self',))
    def calculate_update(self, grads, epoch):
        vd = self.vd or jax.tree_multimap(lambda g: jnp.zeros_like(g), grads)
        # exponentially weighted average of the gradients
        self.vd = jax.tree_multimap(lambda vd, g: self.momentum * vd + (1 - self.momentum) * g, vd, grads)
        return self.vd
