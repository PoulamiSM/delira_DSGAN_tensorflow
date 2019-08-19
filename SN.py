import tensorflow as tf
from keras import backend as K
import numpy as np

from tensorflow.python.framework import ops
from tensorflow.python.keras import backend as K
from tensorflow.python.ops import math_ops
from tensorflow.python.keras.losses import Loss


class spectral(tf.keras.constraints.Constraint):
    def __init__(self, iteration=1, eps=1e-12):
        self.iteration = iteration
        self.eps = eps

    def __call__(self, w):
        w_shape = w.shape.as_list()
        w = K.reshape(w, [-1, w_shape[-1]])

        u = tf.random.truncated_normal([1, w_shape[-1]])

        u_hat = u
        v_hat = None

        for i in range(self.iteration):
            v_ = tf.matmul(u_hat, tf.transpose(w))
            v_hat = v_ / (K.sum(v_ ** 2) ** 0.5 + self.eps)

            u_ = tf.matmul(v_hat, w)
            u_hat = u_ / (K.sum(u_ ** 2) ** 0.5 + self.eps)

        sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))
        w_norm = w / sigma
        w_norm = tf.reshape(w_norm, w_shape)

        return w_norm

    def get_config(self):
        return {'iteration': self.iteration}


class KL_loss(Loss):
    def call(self, y_true, y_pred):
        y_pred = ops.convert_to_tensor(y_pred)
        y_true = math_ops.cast(y_true, y_pred.dtype)
        return tf.keras.metrics.kullback_leibler_divergence(y_true, y_pred)


class JS_loss(Loss):
    '''
            Implementation of pairwise `jsd` based on
            https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence
            https://www.mothur.org/wiki/Jensen-Shannon_Divergence
    '''

    def call(self, y_true, y_pred):
        y_pred = ops.convert_to_tensor(y_pred)
        y_true = math_ops.cast(y_true, y_pred.dtype)

        return self.jsd(y_true, y_pred)

    def jsd(self, y_true, y_pred):
        # normalize
        y_pred = self.norm(y_pred)
        y_true = self.norm(y_true)

        m = y_true + y_pred
        m = math_ops.scalar_mul(0.5, m)

        entropy_pred = tf.keras.metrics.kullback_leibler_divergence(y_pred, m)
        entropy_true = tf.keras.metrics.kullback_leibler_divergence(y_true, m)

        metric = entropy_pred + entropy_true
        metric = math_ops.scalar_mul(0.5, metric)

        return metric

    def norm(self, x):
        norm = math_ops.reduce_sum(x)
        norm = 1 / norm
        x = math_ops.scalar_mul(norm, x)
        return x


