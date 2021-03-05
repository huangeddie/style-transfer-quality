import tensorflow as tf
import tensorflow_probability as tfp
from absl import flags

from distributions import compute_wass_dist, reshape_to_feats

FLAGS = flags.FLAGS


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


class CovarLoss(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        feats1, feats2 = reshape_to_feats(y_true, y_pred)

        mu1 = tf.reduce_mean(feats1, axis=1, keepdims=True)
        mu2 = tf.reduce_mean(feats2, axis=1, keepdims=True)

        mean_loss = tf.reduce_mean((mu1 - mu2) ** 2, axis=1)

        centered_feats1 = feats1 - mu1
        centered_feats2 = feats2 - mu2

        n = tf.cast(tf.shape(centered_feats1)[1], tf.float32)
        covar1 = tf.einsum('bnc,bnd->bcd', centered_feats1, centered_feats1) / n
        covar2 = tf.einsum('bnc,bnd->bcd', centered_feats2, centered_feats2) / n

        covar_loss = tf.reduce_mean((covar1 - covar2) ** 2, axis=[1, 2])

        return mean_loss + covar_loss


class GramianLoss(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        tf.debugging.assert_rank(y_true, 4)
        imshape = tf.shape(y_true)
        num_locs = tf.cast(imshape[1] * imshape[2], y_true.dtype)

        gram_true = tf.linalg.einsum('bijc,bijd->bcd', y_true, y_true) / num_locs
        gram_pred = tf.linalg.einsum('bijc,bijd->bcd', y_pred, y_pred) / num_locs

        return (gram_true - gram_pred) ** 2


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


class WassLoss(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        wass_dist = compute_wass_dist(y_true, y_pred, p=2)
        return wass_dist


class CoWassLoss(tf.keras.losses.Loss):
    def __init__(self, total_steps, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.total_steps = tf.constant(total_steps, tf.float32)
        self.curr_step = tf.constant(0, tf.float32)

    def call(self, y_true, y_pred):
        wass_loss = compute_wass_dist(y_true, y_pred, p=2)

        feats1, feats2 = reshape_to_feats(y_true, y_pred)
        covar1 = tfp.stats.covariance(feats1, sample_axis=1)
        covar2 = tfp.stats.covariance(feats2, sample_axis=1)
        covar_loss = (covar1 - covar2) ** 2

        tf.assert_rank(wass_loss, 2)
        tf.assert_rank(covar_loss, 3)

        alpha = self.curr_step / self.total_steps
        loss = alpha * tf.reduce_mean(wass_loss, axis=1) + tf.reduce_mean(covar_loss, axis=[1, 2])

        self.curr_step.assign_add(tf.ones_like(self.curr_step))
        return loss


loss_dict = {'m1': FirstMomentLoss(), 'm2': SecondMomentLoss(), 'covar': CovarLoss(), 'gram': GramianLoss(),
             'm3': ThirdMomentLoss(), 'wass': WassLoss(), 'cowass': CoWassLoss(FLAGS.train_steps)}
