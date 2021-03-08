import tensorflow as tf
from absl import flags
from sklearn import decomposition

FLAGS = flags.FLAGS


class Preprocess(tf.keras.layers.Layer):
    def __init__(self, preprocess_fn, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.preprocess = preprocess_fn

    def call(self, inputs, **kwargs):
        return self.preprocess(inputs)


class Standardize(tf.keras.layers.Layer):
    def __init__(self, shift=True, scale=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.shift, self.scale = shift, scale

    def build(self, input_shape):
        feat_dim = input_shape[-1]
        self.mean = self.add_weight('mean', [1, 1, 1, feat_dim], trainable=False, initializer='zeros')
        self.variance = self.add_weight('variance', [1, 1, 1, feat_dim], trainable=False, initializer='ones')
        self.configured = self.add_weight('configured', [], trainable=False, dtype=tf.bool, initializer='zeros')

    def configure(self, feats):
        # Precision errors with float32
        feats = tf.cast(feats, tf.float64)

        if self.shift:
            self.mean.assign(tf.cast(tf.reduce_mean(feats, axis=[0, 1, 2], keepdims=True), tf.float32))

        if self.scale:
            self.variance.assign(tf.cast(tf.math.reduce_variance(feats, axis=[0, 1, 2], keepdims=True), tf.float32))

    def call(self, inputs, **kwargs):
        if self.configured == tf.zeros_like(self.configured):
            self.configure(inputs)
            self.configured.assign(tf.ones_like(self.configured))
        return (inputs - self.mean) * tf.math.rsqrt(self.variance + 1e-5)


class PCA(tf.keras.layers.Layer):
    def __init__(self, out_dim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.out_dim = out_dim

    def build(self, input_shape):
        feat_dim = input_shape[-1]
        self.mean = self.add_weight('mean', [1, 1, 1, feat_dim], trainable=False)
        self.projection = self.add_weight('projection', [feat_dim, self.out_dim], trainable=False)

    def configure(self, feats):
        # Precision errors with float32
        feats = tf.cast(feats, tf.float64)
        self.mean.assign(tf.cast(tf.reduce_mean(feats, axis=[0, 1, 2], keepdims=True), tf.float32))

        pca = decomposition.PCA(n_components=self.out_dim, whiten=FLAGS.whiten)
        feats_shape = tf.shape(feats)
        n_samples, feat_dim = tf.reduce_prod(feats_shape[:-1]), feats_shape[-1]

        pca.fit(tf.reshape(feats, [n_samples, feat_dim]))
        tf.debugging.assert_equal(tf.squeeze(self.mean), tf.constant(pca.mean_, dtype=tf.float32))
        self.projection.assign(tf.constant(pca.components_.T, dtype=self.projection.dtype))

    def call(self, inputs, **kwargs):
        x = inputs - self.mean
        components = tf.einsum('bhwc,cd->bhwd', x, self.projection)
        return tf.concat([inputs, components], axis=-1)


class FastICA(tf.keras.layers.Layer):
    def __init__(self, out_dim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.out_dim = out_dim

    def build(self, input_shape):
        feat_dim = input_shape[-1]
        self.mean = self.add_weight('mean', [1, 1, 1, feat_dim], trainable=False)
        self.projection = self.add_weight('projection', [feat_dim, self.out_dim], trainable=False)

    def configure(self, feats):
        # Precision errors with float32
        feats = tf.cast(feats, tf.float64)
        self.mean.assign(tf.cast(tf.reduce_mean(feats, axis=[0, 1, 2], keepdims=True), tf.float32))

        ica = decomposition.FastICA(n_components=self.out_dim)
        feats_shape = tf.shape(feats)
        n_samples, feat_dim = tf.reduce_prod(feats_shape[:-1]), feats_shape[-1]

        ica.fit(tf.reshape(feats, [n_samples, feat_dim]))
        tf.debugging.assert_equal(tf.squeeze(self.mean), tf.constant(ica.mean_, dtype=tf.float32))
        self.projection.assign(tf.constant(ica.components_.T, dtype=self.projection.dtype))

    def call(self, inputs, **kwargs):
        x = inputs - self.mean
        components = tf.einsum('bhwc,cd->bhwd', x, self.projection)
        return tf.concat([inputs, components], axis=-1)
