from abc import abstractmethod


class Optimizer:
    @abstractmethod
    def optimize(self, params, x, y, loss_f):
        pass
