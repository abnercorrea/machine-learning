import tensorflow as tf

from abnercorrea.tensorflow.util.data_prep import split_train_validation_tf, prepend_col
from abnercorrea.tensorflow.util.stat import r2_score, mean_squared_error
from abnercorrea.tensorflow.util.tensorflow import tf_default_device


class RidgeRegression:
    """
    Ordinary LinearRegression = RidgeRegression for alpha = 0
    """
    def __init__(self, algorithm='lu_factorization', device=None):
        assert algorithm in ['inverse_matrix', 'lu_factorization']
        self.w = None
        self.alpha = None
        self.alpha_scores = None
        self.algorithm = algorithm
        self.device = device or tf_default_device()

    def fit(self, X, y, alphas, folds=10):
        self.alpha, self.w, self.alpha_scores = self.fit_tf(X, y, alphas, folds)

    @tf.function(
        input_signature=[
                         tf.TensorSpec(shape=None, dtype=tf.float64),
                         tf.TensorSpec(shape=None, dtype=tf.float64),
                         tf.TensorSpec(shape=None, dtype=tf.float64),
                         tf.TensorSpec(shape=(), dtype=tf.int32),
                         ]
                 )
    def fit_tf(self, X, y, alphas, folds):
        """
        Trains model using cross validation.
        TF based implementation is faster than numpy.
        """
        alphas_size = tf.size(alphas)
        scores = tf.TensorArray(tf.float64, size=alphas_size)
        scores = scores.unstack(tf.zeros_like(alphas))

        for fold in tf.range(folds, dtype=tf.int32):
            # splits train and validation sets
            xtr, ytr, xvl, yvl = split_train_validation_tf(X, y, fold, folds)
            for i in tf.range(alphas_size):
                # fits model using trainig set
                w = self.solve_w(xtr, ytr, alphas[i])
                # calculates score using validation set
                score = self.score_tf(xvl, yvl, w)
                # accumulates score of each alpha over all folds
                scores = scores.write(i, scores.read(i) + score)

        alpha_scores = scores.stack() / folds
        alpha = alphas[tf.argmax(alpha_scores)]
        # trains with best alpha using all train data
        w = self.solve_w(X, y, alpha)
        return alpha, w, alpha_scores

    @tf.function(
        input_signature=[
                         tf.TensorSpec(shape=None, dtype=tf.float64),
                         tf.TensorSpec(shape=None, dtype=tf.float64),
                         tf.TensorSpec(shape=(), dtype=tf.float64)
                         ]
                 )
    def solve_w(self, X, y, alpha):
        """
        Closed form solution.
        w = inv(X.T X + alpha * I) * X.T y
        Uses inverse matrix or solves system of linear eqs.
        """
        print('tracing fit')
        with tf.device(self.device):
            # tf.print(tf.shape(X)[0])
            x = prepend_col(X, 1)
            xt = tf.transpose(x)
            # covariance matrix
            A = xt @ x
            I = tf.eye(tf.shape(xt)[0])
            if self.algorithm == 'inverse_matrix':
                w = tf.linalg.inv(A + alpha * I) @ xt @ y
            elif self.algorithm == 'lu_factorization':
                # Uses LU factorization.
                w = tf.linalg.solve(A + alpha * I, xt @ y)
            return w

    def predict(self, X):
        self.assert_trained()
        return self.predict_tf(X, self.w)

    @tf.function(
        input_signature=[
                         tf.TensorSpec(shape=None, dtype=tf.float64),
                         tf.TensorSpec(shape=None, dtype=tf.float64)
                         ]
                 )
    def predict_tf(self, X, w):
        print('tracing predict')
        with tf.device(self.device):
            x = prepend_col(X, 1)
            predictions = x @ w
            return predictions

    def score(self, X, y):
        self.assert_trained()
        return self.score_tf(X, y, self.w)

    @tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.float64), tf.TensorSpec(shape=None, dtype=tf.float64), tf.TensorSpec(shape=None, dtype=tf.float64)])
    def score_tf(self, X, y, w):
        print('tracing score')
        with tf.device(self.device):
            predictions = self.predict_tf(X, w)
            return r2_score(y, predictions)

    def error(self, X, y):
        self.assert_trained()
        return self.error_tf(X, y, self.w)

    @tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.float64), tf.TensorSpec(shape=None, dtype=tf.float64), tf.TensorSpec(shape=None, dtype=tf.float64)])
    def error_tf(self, X, y, w):
        with tf.device(self.device):
            predictions = self.predict_tf(X, w)
            return mean_squared_error(y, predictions)

    def assert_trained(self):
        assert self.w is not None, 'Please train the model with fit(X, y) before making predictions.'
