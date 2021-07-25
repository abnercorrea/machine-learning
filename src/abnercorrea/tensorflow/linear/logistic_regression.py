import tensorflow as tf

from abnercorrea.tensorflow.util.data_prep import split_train_validation_tf
from abnercorrea.tensorflow.util.stat import logistic_sigmoid


class LogisticRegressionClassifierTF:
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
        self.w, self.alpha, self.alpha_scores = self.fit_tf(X_, y, alphas, folds, self.max_iter, self.tol)

    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=None, dtype=tf.float64),
            tf.TensorSpec(shape=None, dtype=tf.float64),
            tf.TensorSpec(shape=None, dtype=tf.float64),
            tf.TensorSpec(shape=(), dtype=tf.int32),
            tf.TensorSpec(shape=(), dtype=tf.int32),
            tf.TensorSpec(shape=(), dtype=tf.float64),
        ]
    )
    def fit_tf(self, X_, y, alphas, folds, max_iter, tol):
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
                w = self.optimizer(xtr, ytr, alphas[i], max_iter=max_iter, tol=tol)
                # calculates score using validation set
                score = self.score_tf(xvl, yvl, w)
                # accumulates score of each alpha over all folds
                scores = scores.write(i, scores.read(i) + score)

        alpha_scores = scores.stack() / folds
        alpha = alphas[tf.argmax(alpha_scores)]
        # trains with best alpha using all train data
        w = self.optimizer(X_, y, alpha, max_iter=max_iter, tol=tol)
        return w, alpha, alpha_scores

    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=None, dtype=tf.float64),
            tf.TensorSpec(shape=None, dtype=tf.float64),
            tf.TensorSpec(shape=(), dtype=tf.float64),
            tf.TensorSpec(shape=(), dtype=tf.int32),
            tf.TensorSpec(shape=(), dtype=tf.float64),
        ]
    )
    def newton_raphson(self, X_, y, alpha, max_iter, tol):
        """
        This maximizes a penalized log-likelihood(w)

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
        """
        f = tf.shape(X_)[1]
        # w = 0 is a good starting value for the iterative procedure
        w0 = tf.zeros(shape=[f], dtype=tf.float64)
        # since gradient is used to check convergence, this guarantees at least 1 iteration.
        gradient0 = tf.fill(dims=[f], value=tol * 2)

        # checks convergence (gradient = 0)
        not_converged = lambda w, gradient: tf.reduce_any(tf.abs(gradient) >= tol)

        # Newton step
        def newton_step(w_old, gradient_old):
            gradient, p = self.gradient(X_, y, w_old, alpha)
            hessian = self.hessian(X_, p, alpha)
            hessian_inv = tf.linalg.inv(hessian)
            # w_new = w_old - inverse(hessian(log_likelihood(w))) @ gradient(log_likelihood(w))
            w = w_old - hessian_inv @ gradient
            # sets shape of loop variables to prevent errors related to varying shape across iterations.
            w.set_shape(w_old.get_shape())
            gradient.set_shape(gradient_old.get_shape())
            return [w, gradient]

        # since back_prop is not needed, using tf.stop_gradient to prevent gradient computation.
        [w, _] = tf.nest.map_structure(
            tf.stop_gradient,
            tf.while_loop(
                cond=not_converged,
                body=newton_step,
                loop_vars=[w0, gradient0],
                maximum_iterations=max_iter
            )
        )

        return w

    def gradient_descent(self, X_, y, w_start, alpha, max_iter):
        # TODO: hehe
        pass

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
        Returns probability of class 0
        """
        xw = X_ @ w
        return logistic_sigmoid(xw)

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
