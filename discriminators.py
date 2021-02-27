import tensorflow as tf


class BatchNormDiscriminator(tf.keras.layers.Layer):
    def call(self, inputs):
        feats1, feats2 = inputs

        mu1 = tf.reduce_mean(feats1, axis=1)
        mu2 = tf.reduce_mean(feats2, axis=1)

        std1 = tf.math.reduce_std(feats1, axis=1)
        std2 = tf.math.reduce_std(feats2, axis=1)

        self.add_loss((mu1 - mu2) ** 2 + (std1 - std2) ** 2)

        return inputs


class GramianDiscriminator(tf.keras.layers.Layer):
    pass


class ThirdMomentDiscriminator(tf.keras.layers.Layer):
    pass
