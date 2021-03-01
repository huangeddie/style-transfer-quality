import tensorflow as tf

from style_content import FLAGS


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


class SecondMomentLoss(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        feats1, feats2 = y_true, y_pred

        mu1, var1 = tf.nn.moments(feats1, axes=[1, 2], keepdims=True)
        mu2, var2 = tf.nn.moments(feats2, axes=[1, 2], keepdims=True)

        loss = (mu1 - mu2) ** 2 + (var1 - var2) ** 2

        return loss


class ThirdMomentLoss(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        feats1, feats2 = y_true, y_pred

        mu1, var1 = tf.nn.moments(feats1, axes=[1, 2], keepdims=True)
        mu2, var2 = tf.nn.moments(feats2, axes=[1, 2], keepdims=True)

        z1 = (feats1 - mu1) * tf.math.rsqrt(var1 + 1e-3)
        z2 = (feats2 - mu2) * tf.math.rsqrt(var2 + 1e-3)

        skew1 = tf.reduce_mean(z1 ** 3, axis=[1, 2], keepdims=True)
        skew2 = tf.reduce_mean(z2 ** 3, axis=[1, 2], keepdims=True)

        loss = (mu1 - mu2) ** 2 + (var1 - var2) ** 2 + (skew1 - skew2) ** 2

        return loss


class GramianLoss(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        num_locs = tf.cast(tf.shape(y_true)[1], y_true.dtype)

        gram_true = tf.linalg.einsum('bijc,bijd->bcd', y_true, y_true) / num_locs
        gram_pred = tf.linalg.einsum('bijc,bijd->bcd', y_pred, y_pred) / num_locs

        return (gram_true - gram_pred) ** 2


def make_discriminator():
    if FLAGS.disc == 'm2':
        return SecondMomentLoss()
    elif FLAGS.disc == 'gram':
        return GramianLoss()
    elif FLAGS.disc == 'm3':
        return ThirdMomentLoss()
    else:
        raise ValueError(f'unknown discriminator: {FLAGS.disc}')
