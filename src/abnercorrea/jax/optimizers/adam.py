from absl import logging
from functools import partial

import jax

from abnercorrea.jax.optimizers.gradient_descent import MiniBatchSGD


class Adam(MiniBatchSGD):
    """
    Adaptive moment estimation. (Adam = Momentum + RMSProp)

    https://arxiv.org/pdf/1412.6980.pdf
    """
    def __init__(self, momentum=0.9, rms_momentum=0.999, rms_eps=1e-8, **kwargs):
        super().__init__(**kwargs)

        self.momentum = momentum
        self.rms_momentum = rms_momentum
        self.rms_eps = rms_eps
        self.vd = None
        self.sd = None

    @partial(jax.jit, static_argnames=('self',))
    def calculate_update(self, grads, epoch):
        logging.debug(f'epoch: {epoch}')

        zeroes = jax.tree_multimap(lambda g: jnp.zeros_like(g), grads)

        # Applies bias corrections
        vd = jax.tree_multimap(lambda vd, g: self.momentum * vd + (1 - self.momentum) * g, self.vd or zeroes, grads)
        vd_correction = 1 / (1 - self.momentum ** epoch)
        self.vd = jax.tree_multimap(lambda vd: vd * vd_correction, vd)

        sd = jax.tree_multimap(lambda sd, g: self.rms_momentum * sd + (1 - self.rms_momentum) * g ** 2, self.sd or zeroes, grads)
        sd_correction = 1 / (1 - self.rms_momentum ** epoch)
        self.sd = jax.tree_multimap(lambda sd: sd * sd_correction, sd)

        # ADAM update
        update = jax.tree_multimap(lambda vd, sd: vd / (sd ** 0.5 + self.rms_eps), self.vd, self.sd)

        logging.debug(f'vd: {vd}\nsd: {sd}')
        logging.debug(f'vd correction: {vd_correction} - sd correction: {sd_correction}')
        logging.debug(f'vd corrected: {self.vd}\nsd corrected: {self.sd}')
        logging.debug(f'update: {update}')

        return update
