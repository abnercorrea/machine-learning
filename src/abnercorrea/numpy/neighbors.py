import numpy as np

from abnercorrea.numpy.util.data_prep import split_train_validation
from abnercorrea.numpy.util.math import weighted_mode


class KNearestNeighborsClassifier:
    """
    Introduces preditiction of multiple points using matrix operations (no iteration)
    Runs into OOM if too many data points are used. (need to implement batching in "predict")
    """
    def __init__(self, k=None):
        self.k = k
        self.k_scores = None
        self.xtr = None
        self.ytr = None

    def fit(self, X, y, alphas, folds=10):
        """
        Uses cross validation to determine best K.
        """
        alpha_scores = np.zeros(alphas.shape[0], dtype=np.float64)

        for fold in range(folds):
            xtr, ytr, xvl, yvl = split_train_validation(X, y, fold, folds)
            for i in range(alphas.shape[0]):
                self.xtr, self.ytr, self.k = xtr, ytr, alphas[i]
                score = self.score(xvl, yvl)
                alpha_scores[i] += score

        self.k = alphas[alpha_scores.argmax()]
        self.k_scores = alpha_scores / folds
        self.xtr = X
        self.ytr = y

    def predict(self, X):
        """
        Brute force k nearest neighbors. (calculates distances to all training points)
        """
        xtr, ytr, k = self.xtr, self.ytr, self.k
        # squared distances, between row vectors of a  and row vectors of b, can be calculated with:
        # distances = dot(a, a) - 2 * dot(a, b) + dot(b, b)
        # Brilliant... (a - b)^2 = a^2 - 2ab + b^2
        # Sqrt is NOT needed... brilliant
        # First term bradcast as a column, thrid term broadcast as a row
        # distances.shape = (xtr.shape[0], X.shape[0])
        # einsum implementation seems to be faster than above, not by much though.
        # distances = np.einsum('ij,ij->i', xtr, xtr)[:, np.newaxis] - 2 * np.dot(xtr, X.T) + np.einsum('ij,ij->i', X, X)[np.newaxis, :]
        distances = (xtr * xtr).sum(axis=1)[:, np.newaxis] - 2 * np.dot(xtr, X.T) + (X * X).sum(axis=1)[np.newaxis, :]
        # Using argpartition and array slice to get k nearest neighbors indices
        nearest_neighbors_index = distances.T.argpartition(kth=k)[:, :k]
        # k nearest neighbors distances
        nearest_neighbors_dist = np.take_along_axis(distances.T, nearest_neighbors_index, axis=1)
        voting_weights = self.calculate_weights(nearest_neighbors_dist)
        # k nearest neighbors labels
        labels = np.take_along_axis(ytr[np.newaxis, :], nearest_neighbors_index, axis=1)
        # finds the label with max weight sum
        predictions = weighted_mode(a=labels, w=voting_weights, axis=1)[0][:, 0]
        return predictions

    def score(self, X, y):
        """
        Score used is the prediction accuracy
        """
        predictions = self.predict(X)
        n = y.shape[0]
        accurate_count = np.sum(predictions == y, dtype=np.float64)
        accuracy = accurate_count / n
        return accuracy

    def calculate_weights(self, dist):
        """
        if user attempts to classify a point that was zero distance from one
        or more training points, those training points are weighted as 1.0
        and the other points as 0.0
        """
        # ignores division by zero
        with np.errstate(divide='ignore'):
            dist = 1. / dist
        # mask for infinity values
        inf_mask = np.isinf(dist)
        # rows with infinity values
        inf_row = np.any(inf_mask, axis=1)
        # replaces all "inf" with 1. and all other values with 0.
        dist[inf_row] = inf_mask[inf_row]
        return dist
