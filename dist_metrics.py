import tensorflow as tf


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