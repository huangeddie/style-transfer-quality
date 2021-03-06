import tensorflow as tf

from distributions import compute_wass_dist, compute_raw_m2_loss


class MeanLoss(tf.keras.metrics.Metric):
    def __init__(self, name="mean_loss", **kwargs):
        super().__init__(name=name, **kwargs)
        self.mean_loss = self.add_weight(name="mean_loss", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        mu1 = tf.reduce_mean(y_true, axis=[1, 2], keepdims=True)
        mu2 = tf.reduce_mean(y_pred, axis=[1, 2], keepdims=True)

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
        var1 = tf.math.reduce_variance(y_true, axis=[1, 2], keepdims=True)
        var2 = tf.math.reduce_variance(y_pred, axis=[1, 2], keepdims=True)

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


class CovarLoss(tf.keras.metrics.Metric):
    def __init__(self, name="covar_loss", **kwargs):
        super().__init__(name=name, **kwargs)
        self.covar_loss = self.add_weight(name="covar_loss", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        mu1 = tf.reduce_mean(y_true, axis=[1, 2], keepdims=True)
        mu2 = tf.reduce_mean(y_pred, axis=[1, 2], keepdims=True)

        centered_y1 = y_true - mu1
        centered_y2 = y_pred - mu2

        covar_loss = compute_raw_m2_loss(centered_y1, centered_y2)

        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, "float32")
            covar_loss = tf.multiply(covar_loss, sample_weight)

        self.covar_loss.assign_add(tf.reduce_mean(covar_loss))

    def result(self):
        return self.covar_loss

    def reset_states(self):
        # The state of the metric will be reset at the start of each epoch.
        self.covar_loss.assign(0.0)


class GramLoss(tf.keras.metrics.Metric):
    def __init__(self, name="gram_loss", **kwargs):
        super().__init__(name=name, **kwargs)
        self.gram_loss = self.add_weight(name="gram_loss", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        gram_loss = compute_raw_m2_loss(y_true, y_pred)

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
        mu1, var1 = tf.nn.moments(y_true, axes=[1, 2], keepdims=True)
        mu2, var2 = tf.nn.moments(y_pred, axes=[1, 2], keepdims=True)

        z1 = (y_true - mu1) * tf.math.rsqrt(var1 + 1e-3)
        z2 = (y_pred - mu2) * tf.math.rsqrt(var2 + 1e-3)

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


class WassDist(tf.keras.metrics.Metric):
    def __init__(self, name="wass_dist", **kwargs):
        super().__init__(name=name, **kwargs)
        self.wass_dist = self.add_weight(name="wass_dist", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        wass_dist = compute_wass_dist(y_pred, y_true)

        self.wass_dist.assign_add(tf.reduce_mean(wass_dist))

    def result(self):
        return self.wass_dist

    def reset_states(self):
        # The state of the metric will be reset at the start of each epoch.
        self.wass_dist.assign(0.0)
