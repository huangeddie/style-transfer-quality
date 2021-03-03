import tensorflow as tf
from scipy import stats


class MeanLoss(tf.keras.metrics.Metric):
    def __init__(self, name="mean_loss", **kwargs):
        super().__init__(name=name, **kwargs)
        self.mean_loss = self.add_weight(name="mean_loss", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        feats1, feats2 = y_true, y_pred

        mu1 = tf.reduce_mean(feats1, axis=[1, 2], keepdims=True)
        mu2 = tf.reduce_mean(feats2, axis=[1, 2], keepdims=True)

        mean_loss = (mu1 - mu2) ** 2

        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, "float32")
            mean_loss = tf.multiply(mean_loss, sample_weight)

        self.mean_loss.assign_add(tf.reduce_mean(mean_loss))

    def result(self):
        return self.mean_loss

    def reset_states(self):
        # The state of the metric will be reset at the start of each epoch.
        self.mean_loss.assign(0.0)


class VarLoss(tf.keras.metrics.Metric):
    def __init__(self, name="var_loss", **kwargs):
        super().__init__(name=name, **kwargs)
        self.var_loss = self.add_weight(name="var_loss", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        feats1, feats2 = y_true, y_pred

        var1 = tf.math.reduce_variance(feats1, axis=[1, 2], keepdims=True)
        var2 = tf.math.reduce_variance(feats2, axis=[1, 2], keepdims=True)

        var_loss = (var1 - var2) ** 2

        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, "float32")
            var_loss = tf.multiply(var_loss, sample_weight)

        self.var_loss.assign_add(tf.reduce_mean(var_loss))

    def result(self):
        return self.var_loss

    def reset_states(self):
        # The state of the metric will be reset at the start of each epoch.
        self.var_loss.assign(0.0)


class GramLoss(tf.keras.metrics.Metric):
    def __init__(self, name="gram_loss", **kwargs):
        super().__init__(name=name, **kwargs)
        self.gram_loss = self.add_weight(name="gram_loss", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        feats1, feats2 = y_true, y_pred
        tf.debugging.assert_rank(y_true, 4)
        imshape = tf.shape(y_true)
        num_locs = tf.cast(imshape[1] * imshape[2], y_true.dtype)

        gram1 = tf.einsum('bhwc,bhwd->bcd', feats1, feats1) / num_locs
        gram2 = tf.einsum('bhwc,bhwd->bcd', feats2, feats2) / num_locs

        gram_loss = (gram1 - gram2) ** 2

        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, "float32")
            gram_loss = tf.multiply(gram_loss, sample_weight)

        self.gram_loss.assign_add(tf.reduce_mean(gram_loss))

    def result(self):
        return self.gram_loss

    def reset_states(self):
        # The state of the metric will be reset at the start of each epoch.
        self.gram_loss.assign(0.0)


class SkewLoss(tf.keras.metrics.Metric):
    def __init__(self, name="skew_loss", **kwargs):
        super().__init__(name=name, **kwargs)
        self.skew_loss = self.add_weight(name="skew_loss", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        feats1, feats2 = y_true, y_pred

        mu1, var1 = tf.nn.moments(feats1, axes=[1, 2], keepdims=True)
        mu2, var2 = tf.nn.moments(feats2, axes=[1, 2], keepdims=True)

        z1 = (feats1 - mu1) * tf.math.rsqrt(var1 + 1e-3)
        z2 = (feats2 - mu2) * tf.math.rsqrt(var2 + 1e-3)

        skew1 = tf.reduce_mean(z1 ** 3, axis=[1, 2], keepdims=True)
        skew2 = tf.reduce_mean(z2 ** 3, axis=[1, 2], keepdims=True)

        skew_loss = (skew1 - skew2) ** 2

        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, "float32")
            skew_loss = tf.multiply(skew_loss, sample_weight)

        self.skew_loss.assign_add(tf.reduce_mean(skew_loss))

    def result(self):
        return self.skew_loss

    def reset_states(self):
        # The state of the metric will be reset at the start of each epoch.
        self.skew_loss.assign(0.0)


def py_wass_dist(y_true, y_pred):
    wass_dist = []
    bsz = len(y_true)
    for i in range(bsz):
        wass_dist.append(stats.wasserstein_distance(y_true[i].numpy().flatten(), y_pred[i].numpy().flatten()))
    wass_dist = tf.constant(wass_dist, dtype=tf.float32)

    return wass_dist


class WassDist(tf.keras.metrics.Metric):
    def __init__(self, name="wass_dist", **kwargs):
        super().__init__(name=name, **kwargs)
        self.wass_dist = self.add_weight(name="wass_dist", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        wass_dist = tf.py_function(py_wass_dist, [y_true, y_pred], [tf.float32])

        self.wass_dist.assign_add(tf.reduce_mean(wass_dist))

    def result(self):
        return self.wass_dist

    def reset_states(self):
        # The state of the metric will be reset at the start of each epoch.
        self.wass_dist.assign(0.0)
