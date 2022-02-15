from abc import abstractmethod


class Optimizer():
    @abstractmethod
    def calculate_update(self, grads, epoch):
        pass

    @abstractmethod
    def optimize(self, params, x, y, loss_f):
        pass
