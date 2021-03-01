import tensorflow as tf


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


loss_dict = {'m2': SecondMomentLoss(), 'gram': GramianLoss(), 'm3': ThirdMomentLoss()}
