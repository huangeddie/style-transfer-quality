import tensorflow as tf

from distributions import compute_wass_dist


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
        wass_dist = compute_wass_dist(y_true, y_pred, p=2)
        return wass_dist

class VarWassLoss(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        wass_dist = compute_wass_dist(y_true ** 2, y_pred ** 2, p=2)
        return wass_dist


loss_dict = {'m1': FirstMomentLoss(), 'm2': SecondMomentLoss(), 'gram': GramianLoss(), 'm3': ThirdMomentLoss(),
             'wass': WassLoss(), 'var_wass': VarWassLoss()}
