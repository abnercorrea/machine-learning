import numpy as np

from abner_ml.numpy.util.stat import empirical_covariance, logistic_sigmoid, softmax


class GaussianMixtureClassifier:
    def __init__(self):
        self.pi_ = None
        self.mu_ = None
        self.covariance_ = None
        self.classes_ = None
        self.w0_ = None
        self.w_ = None

    def fit(self, X, y):
        """
        Uses maximum likelihood to learn w
        Assumes same covariance matrix for all classes.

        TODO: add support for different covariance matrices
        """
        classes = np.unique(y.ravel())
        assert classes.shape[0] >= 2, f'At least 2 classes must be provided but found {classes.shape[0]}:\n{classes}'

        n = X.shape[0]
        x = [X[y == label] for label in classes]
        pi = np.array([xk.shape[0] / n for xk in x])
        mu = np.array([xk.mean(axis=0) for xk in x])
        covariance = empirical_covariance(X, y, classes, mu)
        cov_inv = np.linalg.inv(covariance)

        # w and w0 will be used to calculate the posterior P(ck|x)
        if classes.shape[0] == 2:
            # Binary classification
            w = (mu[0] - mu[1]) @ cov_inv
            w0 = -.5 * mu[0] @ cov_inv @ mu[0].T + .5 * mu[1] @ cov_inv @ mu[1].T + np.log(pi[0] / pi[1])
        else:
            # Multi class classification
            w = mu @ cov_inv
            w0 = np.sum(-.5 * w * mu, axis=1) + np.log(pi)

        self.classes_, self.pi_, self.mu_, self.covariance_ = classes, pi, mu, covariance
        # learned parameters
        self.w_, self.w0_ = w, w0

    def predict_proba(self, X):
        w0, w, classes = self.w0_, self.w_, self.classes_

        x_wt = X @ w.T + w0

        if self.classes_.shape[0] == 2:
            # For binary classification, the posterior P(ck|x) is a logistic sigmoid
            prediction_proba = logistic_sigmoid(x_wt)
        else:
            # For more than 2 classes, the posterior P(ck|x) is the softmax function (generalization of logistic sigmoid)
            prediction_proba = softmax(x_wt)

        return prediction_proba

    def predict(self, X):
        prediction_proba = self.predict_proba(X)
        classes = self.classes_

        if classes.shape[0] == 2:
            # For binary classification, prediction_proba is the probability of classes[0]
            predictions = np.full(X.shape[0], classes[0])
            predictions[prediction_proba < .5] = classes[1]
        else:
            predictions = np.vectorize(lambda argmax: classes[argmax])(prediction_proba.argmax(axis=1))

        return predictions

    def score(self, X, y):
        """
        The score used is the accuracy of the model.
        """
        predictions = self.predict(X)
        tp_tn = (predictions == y).sum()
        n = y.shape[0]
        return tp_tn / n
