import numpy as np

from ..util.data_prep import split_train_validation
from ..util.stat import r2_score, mean_squared_error


class RidgeRegression:
    def __init__(self, algorithm='lu_factorization'):
        assert algorithm in ['inverse_matrix', 'lu_factorization']
        self.algorithm = algorithm
        self.w = None
        self.alpha = None
        self.alpha_scores = None

    def fit(self, X, y, alphas, folds=10):
        """
        Uses cross validation to find best alpha in alphas.
        Uses r squared score.
        :param X: N x p + 1 array of observations where the first predictor is set to 1 to accommodate for w0
        :param y: Array with N labels
        :param alphas:
        :param folds:
        :return:
        """
        alpha_scores = np.zeros(alphas.shape[0])

        for fold in range(folds):
            xtr, ytr, xvl, yvl = split_train_validation(X, y, fold, folds)
            for i_alpha in range(alphas.shape[0]):
                self.w = self.solve_w(xtr, ytr, alphas[i_alpha])
                score = self.score(xvl, yvl)
                alpha_scores[i_alpha] += score

        self.alpha_scores = alpha_scores / folds
        self.alpha = alphas[alpha_scores.argmax()]
        self.w = self.solve_w(X, y, self.alpha)

    def solve_w(self, X, y, alpha):
        """
        Closed form solution of ridge LR (L2 norm)
        w = inv(X.T X + alpha * I) * X.T y
        Uses inverse matrix or solves system of linear eqs.

        linalg.solve: scipy/linalg/basic.py
        LU decomposition and solve: scipy/linalg/decomp_lu.py
        TODO: implement LU decomposition
        TODO: implement Gauss Jordan

        :param X: N x p + 1 array of observations where the first predictor is set to 1 to accommodate for w0
        :param y: Array with N labels
        :param alpha:
        """
        # covariance matrix
        A = X.T @ X
        I = np.identity(X.T.shape[0])
        Xy = X.T @ y
        if self.algorithm == 'inverse_matrix':
            w = np.linalg.inv(A + alpha * I) @ Xy
        elif self.algorithm == 'lu_factorization':
            # Scikit-learn uses this
            # This is supposed to be faster and more numerically stable than inv. matrix.
            # Uses LU factorization.
            w = np.linalg.solve(A + alpha * I, Xy).T
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
