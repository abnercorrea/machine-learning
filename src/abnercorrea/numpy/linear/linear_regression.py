import numpy as np

from abnercorrea.numpy.util.stat import r2_score, mean_squared_error


class LinearRegression:
    """
    Ordinary Linear Regression (RidgeRegression for alpha = 0)
    """
    def __init__(self):
        self.w = None

    def fit(self, X, y):
        """
        Calculates closed form solution of ordinary LR
        :param X: N x p + 1 array of observations where the first predictor is set to 1 to accommodate for w0
        :param y: Array with N labels
        :return: Closed form solution of ordinary LR
        """
        w = np.linalg.inv(X.T @ X) @ X.T @ y
        self.w = w
        return w

    def predict(self, X):
        """
        :param X: N x p + 1 array of observations where the first predictor is set to 1 to accommodate for w0
        :return:
        """
        assert self.w is not None, 'Please train the model with fit(X, y) before making predictions.'
        predictions = self.w @ X.T
        return predictions

    def score(self, X, y):
        """
        :param X: N x p + 1 array of observations where the first predictor is set to 1 to accommodate for w0
        :param y: Array with N labels
        :return:
        """
        predictions = self.predict(X)
        return r2_score(y, predictions)

    def error(self, X, y):
        """
        :param X: N x p + 1 array of observations where the first predictor is set to 1 to accommodate for w0
        :param y: Array with N labels
        :return:
        """
        predictions = self.predict(X)
        return mean_squared_error(y, predictions)
