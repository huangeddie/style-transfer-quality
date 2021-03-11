import tensorflow as tf
import tensorflow_addons as tfa

from distributions import compute_wass_dist, compute_raw_m2_loss


def compute_mean_metric(y_true, y_pred):
    mu1 = tf.reduce_mean(y_true, axis=[1, 2], keepdims=True)
    mu2 = tf.reduce_mean(y_pred, axis=[1, 2], keepdims=True)
    mean_loss = tf.reduce_mean((mu1 - mu2) ** 2, axis=1)
    return mean_loss


def compute_var_metric(y_true, y_pred):
    var1 = tf.math.reduce_variance(y_true, axis=[1, 2], keepdims=True)
    var2 = tf.math.reduce_variance(y_pred, axis=[1, 2], keepdims=True)

    var_loss = tf.reduce_mean((var1 - var2) ** 2, axis=1)
    return var_loss


def compute_covar_metric(y_true, y_pred):
    mu1 = tf.reduce_mean(y_true, axis=[1, 2], keepdims=True)
    mu2 = tf.reduce_mean(y_pred, axis=[1, 2], keepdims=True)
    centered_y1 = y_true - mu1
    centered_y2 = y_pred - mu2
    covar_loss = compute_raw_m2_loss(centered_y1, centered_y2)
    covar_loss = tf.reduce_mean(covar_loss, axis=1)
    return covar_loss


def compute_gram_metric(y_true, y_pred):
    gram_loss = compute_raw_m2_loss(y_true, y_pred)
    gram_loss = tf.reduce_mean(gram_loss, axis=1)
    return gram_loss


def compute_skew_metric(y_pred, y_true):
    mu1, var1 = tf.nn.moments(y_true, axes=[1, 2], keepdims=True)
    mu2, var2 = tf.nn.moments(y_pred, axes=[1, 2], keepdims=True)
    z1 = (y_true - mu1) * tf.math.rsqrt(var1 + 1e-3)
    z2 = (y_pred - mu2) * tf.math.rsqrt(var2 + 1e-3)
    skew1 = tf.reduce_mean(z1 ** 3, axis=[1, 2], keepdims=True)
    skew2 = tf.reduce_mean(z2 ** 3, axis=[1, 2], keepdims=True)
    skew_loss = tf.reduce_mean((skew1 - skew2) ** 2, axis=1)
    return skew_loss


def compute_wass_metric(y_pred, y_true):
    wass_dist = compute_wass_dist(y_true, y_pred)
    wass_dist = tf.reduce_mean(wass_dist, axis=1)
    return wass_dist


class MeanLoss(tfa.metrics.MeanMetricWrapper):
    def __init__(self, name="mean_loss", **kwargs):
        super().__init__(compute_mean_metric, name=name, **kwargs)


class VarLoss(tfa.metrics.MeanMetricWrapper):
    def __init__(self, name="var_loss", **kwargs):
        super().__init__(compute_var_metric, name=name, **kwargs)


class CovarLoss(tfa.metrics.MeanMetricWrapper):
    def __init__(self, name="covar_loss", **kwargs):
        super().__init__(compute_covar_metric, name=name, **kwargs)


class GramLoss(tfa.metrics.MeanMetricWrapper):
    def __init__(self, name="gram_loss", **kwargs):
        super().__init__(compute_gram_metric, name=name, **kwargs)


class SkewLoss(tfa.metrics.MeanMetricWrapper):
    def __init__(self, name="skew_loss", **kwargs):
        super().__init__(compute_skew_metric, name=name, **kwargs)


class WassDist(tfa.metrics.MeanMetricWrapper):
    def __init__(self, name="wass_dist", **kwargs):
        super().__init__(compute_wass_metric, name=name, **kwargs)
