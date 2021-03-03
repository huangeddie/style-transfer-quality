import tensorflow as tf


class FirstMomentLoss(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        feats1, feats2 = y_true, y_pred

        mu1 = tf.reduce_mean(feats1, axis=[1, 2], keepdims=True)
        mu2 = tf.reduce_mean(feats2, axis=[1, 2], keepdims=True)

        return (mu1 - mu2) ** 2


class SecondMomentLoss(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        feats1, feats2 = y_true, y_pred

        mu1, var1 = tf.nn.moments(feats1, axes=[1, 2], keepdims=True)
        mu2, var2 = tf.nn.moments(feats2, axes=[1, 2], keepdims=True)

        return (mu1 - mu2) ** 2 + (var1 - var2) ** 2


class ThirdMomentLoss(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        feats1, feats2 = y_true, y_pred

        mu1, var1 = tf.nn.moments(feats1, axes=[1, 2], keepdims=True)
        mu2, var2 = tf.nn.moments(feats2, axes=[1, 2], keepdims=True)

        z1 = (feats1 - mu1) * tf.math.rsqrt(var1 + 1e-3)
        z2 = (feats2 - mu2) * tf.math.rsqrt(var2 + 1e-3)

        skew1 = tf.reduce_mean(z1 ** 3, axis=[1, 2], keepdims=True)
        skew2 = tf.reduce_mean(z2 ** 3, axis=[1, 2], keepdims=True)

        return (mu1 - mu2) ** 2 + (var1 - var2) ** 2 + (skew1 - skew2) ** 2


class GramianLoss(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        tf.debugging.assert_rank(y_true, 4)
        imshape = tf.shape(y_true)
        num_locs = tf.cast(imshape[1] * imshape[2], y_true.dtype)

        gram_true = tf.linalg.einsum('bijc,bijd->bcd', y_true, y_true) / num_locs
        gram_pred = tf.linalg.einsum('bijc,bijd->bcd', y_pred, y_pred) / num_locs

        return (gram_true - gram_pred) ** 2


class WassLoss(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        shape = tf.shape(y_true)
        bsz, num_locs, channels = shape[0], shape[1] * shape[2], shape[-1]
        y_true = tf.reshape(y_true, [bsz, num_locs, channels])
        y_pred = tf.reshape(y_pred, [bsz, num_locs, channels])
        y, x = tf.sort(y_true, axis=1), tf.sort(y_pred, axis=1)
        return (y - x) ** 2


loss_dict = {'m1': FirstMomentLoss(), 'm2': SecondMomentLoss(), 'gram': GramianLoss(), 'm3': ThirdMomentLoss(),
             'wass': WassLoss()}
