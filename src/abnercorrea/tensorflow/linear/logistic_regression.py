import tensorflow as tf

from ..util.data_prep import split_train_validation_tf
from ..util.stat import logistic_sigmoid
from ..util.tensorflow import tf_while_loop_body


class LogisticRegressionClassifierTF:
    def __init__(self, optimizer='Newton-Raphson', max_iter=10, tol=1e-14, learning_rate=1e-2):
        assert optimizer in ['Newton-Raphson', 'sgd'], f'Optimizer {optimizer} not supported.'

        self.optimizer = optimizer
        self.tol = tol
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.w, self.alpha, self.alpha_scores = None, None, None

    def fit(self, X_, y, alphas=None, folds=10):
        params = self.fit_tf(X_, y, alphas, folds, self.max_iter, self.tol, self.learning_rate)

        self.w, self.alpha, self.alpha_scores = params

    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=None, dtype=tf.float64),
            tf.TensorSpec(shape=None, dtype=tf.float64),
            tf.TensorSpec(shape=None, dtype=tf.float64),
            tf.TensorSpec(shape=(), dtype=tf.int32),
            tf.TensorSpec(shape=(), dtype=tf.int32),
            tf.TensorSpec(shape=(), dtype=tf.float64),
            tf.TensorSpec(shape=(), dtype=tf.float64),
        ]
    )
    def fit_tf(self, X_, y, alphas, folds, max_iter, tol, learning_rate):
        """
        Trains model and finds weights w and hyper-parameter alpha (regularization).

        :param y: denotes the vector of yi values (yi = 1 for class 1 and yi = 0 for class 2)
        """
        alphas_size = tf.size(alphas)
        scores = tf.TensorArray(tf.float64, size=alphas_size)
        scores = scores.unstack(tf.zeros_like(alphas))

        for fold in tf.range(folds, dtype=tf.int32):
            # splits train and validation sets
            xtr, ytr, xvl, yvl = split_train_validation_tf(X_, y, fold, folds)
            for i in tf.range(alphas_size):
                # fits model using trainig set
                w = self.optimize(xtr, ytr, alphas[i], max_iter=max_iter, tol=tol, learning_rate=learning_rate)
                # calculates score using validation set
                score = self.score_tf(xvl, yvl, w)
                # accumulates score of each alpha over all folds
                scores = scores.write(i, scores.read(i) + score)

        alpha_scores = scores.stack() / folds
        alpha = alphas[tf.argmax(alpha_scores)]
        # trains with best alpha using all train data
        w = self.optimize(X_, y, alpha, max_iter=max_iter, tol=tol, learning_rate=learning_rate)
        return w, alpha, alpha_scores

    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=None, dtype=tf.float64),
            tf.TensorSpec(shape=None, dtype=tf.float64),
            tf.TensorSpec(shape=(), dtype=tf.float64),
            tf.TensorSpec(shape=(), dtype=tf.int32),
            tf.TensorSpec(shape=(), dtype=tf.float64),
            tf.TensorSpec(shape=(), dtype=tf.float64),
        ]
    )
    def optimize(self, X_, y, alpha, max_iter, tol, learning_rate):
        """
        This maximizes a penalized log-likelihood(w)

        The penalty used is: -.5 * alpha * w^2 (L2)

        We typically do not penalize the intercept term, and standardize the predictors for the penalty to be meaningful.

        It seems that w = 0 is a good starting value for the iterative procedure, although convergence is never guaranteed.
        Typically the algorithm does converge, since the log-likelihood is concave, but overshooting can occur.
        In the rare cases that the log-likelihood decreases, step size halving will guarantee convergence.

        This algorithm is referred to as iteratively reweighted least squares or IRLS.

        TODO: implement step-size halving in case of not converging
        """
        f = tf.shape(X_)[1]
        # w = 0 is a good starting value for the iterative procedure
        w0 = tf.zeros(shape=[f], dtype=tf.float64)
        # since gradient is used to check convergence, this guarantees at least 1 iteration.
        gradient0 = tf.fill(dims=[f], value=tol * 2)

        # checks convergence (gradient = 0 which numerically becomes gradient < tol)
        def not_converged(wi, gradient):
            return tf.reduce_any(tf.abs(gradient) >= tol)

        @tf_while_loop_body()
        def newton_raphson(wi, gradient):
            """
            From HTF:
            The Newton–Raphson algorithm uses the first-derivative or Gradient and the second-derivative or Hessian matrix.

            Staring with w_old, the Newton step is:
            w_new = w_old - inverse(hessian(log_likelihood(w))) @ gradient(log_likelihood(w))
            """
            gradient, p = self.gradient(X_, y, wi, alpha)
            hessian = self.hessian(X_, p, alpha)
            hessian_inv = tf.linalg.inv(hessian)
            # w_new = w - inverse(hessian(log_likelihood(w))) @ gradient(log_likelihood(w))
            wi -= hessian_inv @ gradient
            return [wi, gradient]

        # TODO: SGD
        # @tf_while_loop_body()
        # def sgd(wi, gradient):
        #     """
        #     SGD step calculates the gradient to update the value of w.
        #
        #     Staring with w_old, the Newton step is:
        #     w_new = w_old - learning_rate * gradient(log_likelihood(w))
        #     """
        #     gradient, _ = self.gradient(X_, y, wi, alpha)
        #     # w_new = w_old - learning_rate * gradient(log_likelihood(w))
        #     wi -= learning_rate * gradient
        #     return [wi, gradient]

        # since back_prop is not needed, using tf.stop_gradient to prevent gradient computation.
        [w, _] = tf.nest.map_structure(
            tf.stop_gradient,
            tf.while_loop(
                cond=not_converged,
                body=newton_raphson,
                loop_vars=[w0, gradient0],
                maximum_iterations=max_iter
            )
        )
        return w

    def score(self, X_, y):
        return self.score_tf(X_, y, self.w)

    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=None, dtype=tf.float64),
            tf.TensorSpec(shape=None, dtype=tf.float64),
            tf.TensorSpec(shape=None, dtype=tf.float64),
        ]
    )
    def score_tf(self, X_, y, w):
        """
        Score used is the accuracy of the model. (true positive + true negative rate)
        """
        predictions = self.predict_tf(X_, w)
        accurate = tf.reduce_sum(tf.where(y == predictions, 1., 0.))
        n = tf.size(y)
        accuracy = accurate / n
        return accuracy

    def predict_proba(self, X_):
        return self.predict_proba_tf(X_, self.w)

    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=None, dtype=tf.float64),
            tf.TensorSpec(shape=None, dtype=tf.float64),
        ]
    )
    def predict_proba_tf(self, X_, w):
        """
        Returns probability p of class 1
        If p >= 0.5, then the predicted class is class 1 otherwise it's class 2.
        """
        xw = X_ @ w
        p = logistic_sigmoid(xw)
        return p

    def predict(self, X_):
        return self.predict_tf(X_, self.w)

    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=None, dtype=tf.float64),
            tf.TensorSpec(shape=None, dtype=tf.float64),
        ]
    )
    def predict_tf(self, X_, w):
        """
        Prediction is a dot product... X_ @ w
        Predicted values follow the same convention used to create the y vector:
        1 = Class 1
        0 = Class 2
        """
        xw = X_ @ w
        predictions = tf.where(xw >= 0, 1, 0)
        return predictions

    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=None, dtype=tf.float64),
            tf.TensorSpec(shape=None, dtype=tf.float64),
            tf.TensorSpec(shape=None, dtype=tf.float64),
            tf.TensorSpec(shape=(), dtype=tf.float64),
        ]
    )
    def gradient(self, X_, y, w, alpha):
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

    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=None, dtype=tf.float64),
            tf.TensorSpec(shape=None, dtype=tf.float64),
            tf.TensorSpec(shape=(), dtype=tf.float64),
        ]
    )
    def hessian(self, X_, p, alpha):
        """
        Second-derivative or Hessian matrix of log-likelihood(w).

        Hessian = X.T @ R @ X + alpha * I
        """
        # n: # of data points
        # f: # of features + 1
        x_shape = tf.shape(X_)
        n, f = x_shape[0], x_shape[1]
        # R is a N×N diagonal matrix of weights with ith diagonal element sigmoid(xi; w)(1 − sigmoid(xi; w))
        # The derivative of the sigmoid is sigmoid * (1 - sigmoid)
        sigmoid_deriv = p * (1 - p)
        R = tf.linalg.set_diag(tf.zeros([n, n], dtype=tf.float64), sigmoid_deriv)
        # X_.T @ R @ X_ shape is (f, f)
        H = -X_.T @ R @ X_ - alpha * tf.eye(f)
        return H
