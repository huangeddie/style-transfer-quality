import tensorflow as tf
from absl import flags

from distributions import compute_wass_dist, compute_co_raw_m2_loss, compute_covar_loss, compute_mean_loss, \
    compute_var_loss

FLAGS = flags.FLAGS


class NoOpLoss(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        return tf.zeros(tf.shape(y_true)[0], dtype=y_true.dtype)


class M1Loss(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        return compute_mean_loss(y_true, y_pred, p=2)


class M1M2Loss(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        mean_loss = compute_mean_loss(y_true, y_pred, p=2)
        var_loss = compute_var_loss(y_true, y_pred, p=2)
        return mean_loss + var_loss


class M1CovarLoss(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        mean_loss = compute_mean_loss(y_true, y_pred, p=2)
        covar_loss = compute_covar_loss(y_true, y_pred, p=2)
        return mean_loss + covar_loss


class CoRawM2Loss(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        return compute_co_raw_m2_loss(y_true, y_pred, p=2)


class WassLoss(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        return compute_wass_dist(y_true, y_pred, p=2)


loss_dict = {'m1': M1Loss, 'm1_m2': M1M2Loss, 'm1_covar': M1CovarLoss, 'corawm2': CoRawM2Loss, 'wass': WassLoss,
             None: NoOpLoss}
