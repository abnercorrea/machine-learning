import tensorflow as tf

from abnercorrea.tensorflow.util.data_prep import prepend_col
from abnercorrea.tensorflow.util.stat import r2_score
from abnercorrea.tensorflow.util.tensorflow import tf_default_device


class LinearRegression:
    def __init__(self):
        self.tf_device = tf_default_device()
        self.w = None

    def fit(self, X, y):
        self.w = self.fit_tf(X, y)
        return self.w

    @tf.function
    def fit_tf(self, X, y):
        with tf.device(self.tf_device):
            x = prepend_col(X, 1)
            xt = tf.transpose(x)
            w = tf.linalg.inv(xt @ x) @ xt @ y
            return w

    def predict(self, X):
        self.assert_trained()
        return self.predict_tf(X, self.w)

    @tf.function
    def predict_tf(self, X, w):
        with tf.device(self.tf_device):
            x = prepend_col(X, 1)
            predictions = x @ w
            return predictions

    def score(self, X, y):
        self.assert_trained()
        return self.score_tf(X, y, self.w)

    @tf.function
    def score_tf(self, X, y, w):
        """
        R-squared score
        """
        with tf.device(self.tf_device):
            predictions = self.predict_tf(X, w)
            return r2_score(y, predictions)

    def assert_trained(self):
        assert self.w is not None, 'Please train the model with fit(X, y) before making predictions.'
