import tensorflow as tf
import tensorflow.experimental.numpy as tnp
from tensorflow.python.ops.numpy_ops import np_config

from abner_ml.tensorflow.util.data_prep import split_train_validation_tf
from abner_ml.tensorflow.util.math import weighted_mode

np_config.enable_numpy_behavior()


class KNearestNeighborsClassifierTF:
    """
    Brute force implementation of K nearest neighbors.

    Introduces prediction of multiple points using matrix operations (no iteration)
    Runs into OOM if too many data points are used. (need to implement batching in "predict")
    """
    def __init__(self, k=None, xtr=None, ytr=None):
        self.k = k
        self.xtr = xtr
        self.ytr = ytr
        self.k_scores = None

    def fit(self, X, y, alphas, folds=10):
        k, k_scores = self.fit_tf(X, y, alphas, folds)
        self.xtr, self.ytr = X, y
        self.k, self.k_scores = k.numpy(), k_scores.numpy()

    @tf.function(
        input_signature=[
                         tf.TensorSpec(shape=None, dtype=tf.float64),
                         tf.TensorSpec(shape=None, dtype=tf.float64),
                         tf.TensorSpec(shape=None, dtype=tf.int32),
                         tf.TensorSpec(shape=(), dtype=tf.int32),
                         ]
                 )
    def fit_tf(self, X, y, alphas, folds):
        """
        Trains model using cross validation.
        """
        alphas_size = tf.size(alphas)
        scores = tf.TensorArray(tf.float64, size=alphas_size)
        scores = scores.unstack(tf.zeros_like(alphas, dtype=tf.float64))

        for fold in tf.range(folds, dtype=tf.int32):
            # splits train and validation sets
            xtr, ytr, xvl, yvl = split_train_validation_tf(X, y, fold, folds)
            for i in tf.range(alphas_size):
                # lazy learner...
                alpha = alphas[i]
                # calculates score using validation set
                score = self.score_tf(xvl, yvl, xtr, ytr, alpha)
                # accumulates score of each alpha over all folds
                scores = scores.write(i, scores.read(i) + score)

        alpha_scores = scores.stack() / folds
        # best alpha has the highest score sum over all folds
        alpha = alphas[tf.argmax(alpha_scores)][0]
        return alpha, alpha_scores

    def predict(self, X):
        assert self.xtr is not None and self.ytr is not None and self.k, "k, X and y must be provided at creation or you must use the 'fit' method."
        return self.predict_tf(X, self.xtr, self.ytr, self.k)

    @tf.function(
        input_signature=[
                         tf.TensorSpec(shape=None, dtype=tf.float64),
                         tf.TensorSpec(shape=None, dtype=tf.float64),
                         tf.TensorSpec(shape=None, dtype=tf.float64),
                         tf.TensorSpec(shape=(), dtype=tf.int32),
                         ]
                 )
    def predict_tf(self, X, xtr, ytr, k):
        """
        Brute force k nearest neighbors. (calculates distances to all training points)
        """
        # squared distances, between row vectors of a  and row vectors of b, can be calculated with:
        # distances = dot(a, a) - 2 * dot(a, b) + dot(b, b)
        # Brilliant... (a - b)^2 = a^2 - 2ab + b^2
        # Sqrt is NOT needed... brilliant
        # First term broadcast as a column, third term broadcast as a row
        # distances.shape = (xtr.shape[0], X.shape[0])
        # einsum implementatio seems to be faster than above, not by much though.
        # distances = tnp.einsum('ij,ij->i', xtr, xtr)[:, tnp.newaxis] - 2 * tnp.dot(xtr, X.T) + tnp.einsum('ij,ij->i', X, X)[tnp.newaxis, :]
        distances = tnp.sum(xtr * xtr, axis=1)[:, tnp.newaxis] - 2 * tnp.dot(xtr, X.T) + tnp.sum(X * X, axis=1)[tnp.newaxis, :]
        # Using top_k.indices and -distance to emulate numpy argpartition
        nearest_neighbors_index = tf.math.top_k(-distances.T, k=k).indices
        # Distances to k nearest neighbors
        nearest_neighbors_dist = tnp.take_along_axis(distances.T, nearest_neighbors_index, axis=1)
        voting_weights = self.neighbor_weights(nearest_neighbors_dist)
        # K nearest neighbors labels
        labels = tnp.take_along_axis(ytr[tnp.newaxis, :], nearest_neighbors_index, axis=1)
        # finds the label with max weight sum
        predictions = weighted_mode(a=labels, w=voting_weights, axis=1)[0][:, 0]
        return predictions

    def score(self, X, y):
        return self.score_tf(X, y, self.xtr, self.ytr, self.k)

    @tf.function(
        input_signature=[
                         tf.TensorSpec(shape=None, dtype=tf.float64),
                         tf.TensorSpec(shape=None, dtype=tf.float64),
                         tf.TensorSpec(shape=None, dtype=tf.float64),
                         tf.TensorSpec(shape=None, dtype=tf.float64),
                         tf.TensorSpec(shape=(), dtype=tf.int32),
                         ]
                 )
    def score_tf(self, X, y, xtr, ytr, k):
        predictions = self.predict_tf(X, xtr, ytr, k)
        n = tf.size(y)
        accurate = tnp.sum(predictions == y, dtype=tf.float64)
        accuracy = accurate / n
        return accuracy

    @tf.function(
        input_signature=[
                         tf.TensorSpec(shape=None, dtype=tf.float64),
                         ]
                 )
    def neighbor_weights(self, dist):
        """
        if user attempts to classify a point that was zero distance from one
        or more training points, those training points are weighted as 1.0
        and the other points as 0.0
        """
        inv_dist = 1. / dist
        inf_mask = tf.math.is_inf(inv_dist)
        inf_row_mask = tf.reduce_any(inf_mask, axis=1)
        indices = tf.where(tf.equal(True, inf_row_mask))
        rows_with_inf = tf.cast(tf.gather_nd(tf.where(inf_mask, 1, 0), indices), tf.float64)
        weights = tf.tensor_scatter_nd_update(inv_dist, indices=indices, updates=rows_with_inf)
        return weights
