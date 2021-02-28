import tensorflow as tf

from style_content import FLAGS


class Skewness(tf.keras.metrics.Metric):
    def __init__(self, name="skewness", **kwargs):
        super().__init__(name=name, **kwargs)
        self.skew1 = self.add_weight(name="skew1", initializer="zeros")
        self.skew2 = self.add_weight(name="skew2", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        feats1, feats2 = y_true, y_pred

        mu1 = tf.reduce_mean(feats1, axis=1, keepdims=True)
        mu2 = tf.reduce_mean(feats2, axis=1, keepdims=True)

        std1 = tf.math.reduce_std(feats1, axis=1, keepdims=True)
        std2 = tf.math.reduce_std(feats2, axis=1, keepdims=True)

        z1 = (feats1 - mu1) / (std1 + 1e-5)
        z2 = (feats2 - mu2) / (std2 + 1e-5)

        skew1 = tf.reduce_mean(z1 ** 3, axis=1, keepdims=True)
        skew2 = tf.reduce_mean(z2 ** 3, axis=1, keepdims=True)

        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, "float32")
            skew1 = tf.multiply(skew1, sample_weight)
            skew2 = tf.multiply(skew2, sample_weight)

        self.skew1.assign_add(tf.reduce_mean(skew1))
        self.skew2.assign_add(tf.reduce_mean(skew2))

    def result(self):
        return self.skew1

    def reset_states(self):
        # The state of the metric will be reset at the start of each epoch.
        self.skew1.assign(0.0)


class FirstMomentLoss(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        tf.debugging.assert_rank(y_true, 3)
        tf.debugging.assert_rank(y_pred, 3)

        mu1 = tf.reduce_mean(y_true, axis=1)
        mu2 = tf.reduce_mean(y_pred, axis=1)

        std1 = tf.math.reduce_std(y_true, axis=1)
        std2 = tf.math.reduce_std(y_pred, axis=1)

        loss = (mu1 - mu2) ** 2 + (std1 - std2) ** 2

        return loss


class ThirdMomentLoss(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        feats1, feats2 = y_true, y_pred
        tf.debugging.assert_rank(feats1, 3)
        tf.debugging.assert_rank(feats2, 3)

        mu1 = tf.reduce_mean(feats1, axis=1, keepdims=True)
        mu2 = tf.reduce_mean(feats2, axis=1, keepdims=True)

        std1 = tf.math.reduce_std(feats1, axis=1, keepdims=True)
        std2 = tf.math.reduce_std(feats2, axis=1, keepdims=True)

        z1 = (feats1 - mu1) / (std1 + 1e-5)
        z2 = (feats2 - mu2) / (std2 + 1e-5)

        skew1 = tf.reduce_mean(z1 ** 3, axis=1, keepdims=True)
        skew2 = tf.reduce_mean(z2 ** 3, axis=1, keepdims=True)

        loss = (mu1 - mu2) ** 2 + (std1 - std2) ** 2 + (skew1 - skew2) ** 2

        return loss


class GramianLoss(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        tf.debugging.assert_rank(y_true, 3)
        tf.debugging.assert_rank(y_pred, 3)

        num_locs = tf.cast(tf.shape(y_true)[1], y_true.dtype)

        gram_true = tf.linalg.einsum('bnc,bnd->bcd', y_true, y_true) / num_locs
        gram_pred = tf.linalg.einsum('bnc,bnd->bcd', y_pred, y_pred) / num_locs

        return (gram_true - gram_pred) ** 2


def make_discriminator():
    if FLAGS.disc == 'm1':
        return FirstMomentLoss()
    elif FLAGS.disc == 'gram':
        return GramianLoss()
    elif FLAGS.disc == 'm3':
        return ThirdMomentLoss()
    else:
        raise ValueError(f'unknown discriminator: {FLAGS.disc}')
