import numpy as np

from abnercorrea.numpy.util.data_prep import to_binary_classes, split_train_validation
from abnercorrea.numpy.util.stat import logistic_sigmoid


class LogisticRegressionClassifier:
    def __init__(self, algorithm='Newton-Raphson', max_iter=10, tol=1e-14):
        if algorithm == 'Newton-Raphson':
            self.optimizer = self.newton_raphson
        elif algorithm == 'gradient-descent':
            self.optimizer = self.gradient_descent
        else:
            raise ValueError(f'Algorithm {algorithm} not supported.')

        self.tol = tol
        self.max_iter = max_iter

    def fit(self, X_, y, alphas=None, folds=10):
        """
        Fits best w using optimizer provided and determines best alpha using cross validation.

        :param X_: Predictors are all standardized to have mean zero and unit norm. Predictors must contain an appened column of 1s to accommodate for w0.
        """
        # yb denote the vector of yi values (yi = 1 for class 1 and yi = 0 for class 2)
        yb, self.classes = to_binary_classes(y)

        alpha_scores = np.zeros(len(alphas))

        for fold in range(folds):
            xtr, ytr, xvl, yvl = split_train_validation(X_, yb, fold, folds)
            for i in range(len(alphas)):
                w = self.optimizer(xtr, ytr, alphas[i], max_iter=self.max_iter, tol=self.tol)
                score = self.score(xvl, yvl, w)
                alpha_scores[i] += score

        self.alpha_scores = alpha_scores / folds
        self.alpha = alphas[alpha_scores.argmax()]
        self.w = self.optimizer(X_, yb, self.alpha, max_iter=self.max_iter, tol=self.tol)

    def newton_raphson(self, X_, y, alpha, max_iter=10, tol=1e-14, w_start=None):
        """
        This maximizes a penalized log-likelihood, l(w, alpha)

        l(w, alpha) = sum(yi @ w.T @ xi - log(1 + exp(w.T @ xi))) -.5 * alpha * w^2
        yi = 1 for class 1 and yi = 0 for class 2

        The penalty used is: -.5 * alpha * w^2 (L2)

        We typically do not penalize the intercept term, and standardize the predictors for the penalty to be meaningful.

        From HTF:

        The Newton–Raphson algorithm uses the first-derivative or Gradient and the second-derivative or Hessian matrix.

        Staring with w_old, the Newton step is:

            w_new = w_old - inverse(hessian(log_likelihood(w))) @ gradient(log_likelihood(w))

        It seems that w = 0 is a good starting value for the iterative procedure, although convergence is never guaranteed.
        Typically the algorithm does converge, since the log-likelihood is concave, but overshooting can occur.
        In the rare cases that the log-likelihood decreases, step size halving will guarantee convergence.

        This algorithm is referred to as iteratively reweighted least squares or IRLS.

        TODO: implement step-size halving in case of not converging

        :param X_: Predictors are all standardized to have mean zero and unit norm. Predictors must contain an appened column of 1s to accommodate for w0.
        """
        w = w_start or np.zeros(X_.shape[1])
        iter = 0
        converged = False
        # step_size = 1

        while not converged and iter < max_iter:
            # Newton step
            gradient, p = self.gradient(X_, y, w, alpha)
            hessian = self.hessian(X_, p, alpha)
            hessian_inv = np.linalg.inv(hessian)
            # w_new = w_old - inverse(hessian(log_likelihood(w))) @ gradient(log_likelihood(w))
            w -= hessian_inv @ gradient
            # checks convergence
            converged = np.all(gradient < tol)
            iter += 1

        # print(f'Alpha: {alpha} - Iterations: {iter} - Converged: {converged}')
        return w

    def gradient_descent(self, X_, y, w_start, alpha, max_iter):
        """
        :param X_: Predictors are all standardized to have mean zero and unit norm. Predictors must contain an appened column of 1s to accommodate for w0.
        """
        # TODO: hehe
        pass

    def score(self, X_, y, w=None, y_binary=False):
        """
        Score used is the accuracy of the model. (true positive + true negative rate)

        :param X_: Predictors are all standardized to have mean zero and unit norm. Predictors must contain an appened column of 1s to accommodate for w0.
        """
        predictions = self.predict(X_, w)
        tp_tn_count = np.sum(y == predictions)
        accuracy = tp_tn_count / y.shape[0]
        return accuracy

    def predict_proba(self, X_, w=None):
        """
        Returns probability of class 0

        :param X_: Predictors are all standardized to have mean zero and unit norm. Predictors must contain an appened column of 1s to accommodate for w0.
        """
        w = self.w if w is None else w
        xw = X_ @ w
        return logistic_sigmoid(xw)

    def predict(self, X_, w=None):
        """
        Prediction is a dot product... X_ @ w

        :param X_: Predictors are all standardized to have mean zero and unit norm. Predictors must contain an appened column of 1s to accommodate for w0.
        """
        w = self.w if w is None else w
        xw = X_ @ w
        predictions = np.zeros_like(xw, dtype=np.int8)
        predictions[xw >= 0] = 1
        return predictions

    def gradient(self, X_, y, w, alpha=0):
        """
        First-derivative or Gradient of log-likelihood (w)

        y denote the vector of yi values (yi = 1 for class 1 and yi = 0 for class 2)
        p vector of fitted probabilities p(xi; w)

        Gradient = X.T * (y - p) - alpha * w
        """
        xw = X_ @ w
        # vector of fitted probabilities p(xi; w)
        p = logistic_sigmoid(xw)
        # gradient vector
        gradient = X_.T @ (y - p) - alpha * w
        return gradient, p

    def hessian(self, X_, p, alpha=0):
        """
        Second-derivative or Hessian matrix of log-likelihood(w).

        Hessian = X.T @ R @ X + alpha * I
        """
        # n: # of data points
        # f: # of features + 1
        n, f = X_.shape
        # R is a N×N diagonal matrix of weights with ith diagonal element sigmoid(xi; w)(1 − sigmoid(xi; w))
        # The derivative of the sigmoid is sigmoid * (1 - sigmoid)
        R = np.zeros([n, n])
        np.fill_diagonal(R, p * (1 - p))
        # X_.T @ R @ X_ shape is (f, f)
        H = -X_.T @ R @ X_ - alpha * np.identity(f)
        return H
