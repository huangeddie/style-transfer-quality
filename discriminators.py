import tensorflow as tf


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
    pass


class GramianLoss(tf.keras.losses.Loss):
    pass
