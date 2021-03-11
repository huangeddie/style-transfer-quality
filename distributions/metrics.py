import tensorflow as tf

from distributions import compute_wass_dist, compute_raw_m2_loss


class MeanLoss(tf.keras.metrics.Metric):
    def __init__(self, name="mean_loss", **kwargs):
        super().__init__(name=name, **kwargs)
        self.mean_loss = self.add_weight(name="mean_loss", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        mu1 = tf.reduce_mean(y_true, axis=[1, 2], keepdims=True)
        mu2 = tf.reduce_mean(y_pred, axis=[1, 2], keepdims=True)

        mean_loss = tf.reduce_mean((mu1 - mu2) ** 2, axis=1)

        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, "float32")
            mean_loss = tf.multiply(mean_loss, sample_weight)
        mean_loss = tf.reduce_mean(mean_loss)
        if tf.math.is_finite(mean_loss):
            self.mean_loss.assign_add(mean_loss)

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

        var_loss = tf.reduce_mean((var1 - var2) ** 2, axis=1)

        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, "float32")
            var_loss = tf.multiply(var_loss, sample_weight)

        var_loss = tf.reduce_mean(var_loss)
        if tf.math.is_finite(var_loss):
            self.var_loss.assign_add(var_loss)

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
        covar_loss = tf.reduce_mean(covar_loss, axis=1)

        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, "float32")
            covar_loss = tf.multiply(covar_loss, sample_weight)

        covar_loss = tf.reduce_mean(covar_loss)
        if tf.math.is_finite(covar_loss):
            self.covar_loss.assign_add(covar_loss)

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
        gram_loss = tf.reduce_mean(gram_loss, axis=1)

        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, "float32")
            gram_loss = tf.multiply(gram_loss, sample_weight)

        gram_loss = tf.reduce_mean(gram_loss)
        if tf.math.is_finite(gram_loss):
            self.gram_loss.assign_add(gram_loss)

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

        skew_loss = tf.reduce_mean((skew1 - skew2) ** 2, axis=1)

        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, "float32")
            skew_loss = tf.multiply(skew_loss, sample_weight)

        skew_loss = tf.reduce_mean(skew_loss)
        if tf.math.is_finite(skew_loss):
            self.skew_loss.assign_add(skew_loss)

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
        wass_dist = compute_wass_dist(y_true, y_pred)
        wass_dist = tf.reduce_mean(wass_dist, axis=1)

        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, "float32")
            wass_dist = tf.multiply(wass_dist, sample_weight)

        wass_dist = tf.reduce_mean(wass_dist)
        if tf.math.is_finite(wass_dist):
            self.wass_dist.assign_add(wass_dist)

    def result(self):
        return self.wass_dist

    def reset_states(self):
        # The state of the metric will be reset at the start of each epoch.
        self.wass_dist.assign(0.0)
