import tensorflow as tf
from absl import flags
from absl.testing import absltest

import model as scm
import model.layers

FLAGS = flags.FLAGS


class TestModel(absltest.TestCase):
    def test_model_train_step(self):
        FLAGS(['', '--feat_model=fast'])
        feat_model = scm.make_feat_model([32, 32, 3])

        for sample_size in [None, 64]:
            sc_model = scm.SCModel(feat_model, sample_size=sample_size, loss_warmup=0)
            sc_model.compile('adam',
                             loss={'style': [tf.keras.losses.MeanSquaredError(), tf.keras.losses.MeanSquaredError()]})
            # Random uniform doesn't support uint8
            x = tf.random.uniform([1, 32, 32, 3], maxval=255, dtype=tf.int32)
            y = tf.random.uniform([1, 32, 32, 3], maxval=255, dtype=tf.int32)
            feats = {'style': [tf.random.uniform([1, 16, 16, 3]), tf.random.uniform([1, 8, 8, 3])],
                     'content': [tf.random.uniform([1, 16, 16, 3]), tf.random.uniform([1, 8, 8, 3])]}
            metrics = sc_model.train_step(((x, y), feats))
            self.assertIsInstance(metrics, dict)

    def test_model_call(self):
        FLAGS(['', '--feat_model=fast', '--style_image=out/starry_night.jpg'])
        feat_model = scm.make_feat_model([32, 32, 3])
        sc_model = scm.SCModel(feat_model, sample_size=None, loss_warmup=0)
        # Random uniform doesn't support uint8
        x = tf.random.uniform([1, 32, 32, 3], maxval=255, dtype=tf.int32)
        y = tf.random.uniform([1, 32, 32, 3], maxval=255, dtype=tf.int32)
        _ = sc_model((x, y))

    def test_pca_einsum(self):
        for _ in range(100):
            a = tf.random.normal([8, 32, 32, 64])
            b = tf.random.normal([64, 16])
            einsum_result = tf.einsum('bhwc,cd->bhwd', a, b)

            flat_a = tf.reshape(a, [-1, 64])
            flat_result = tf.matmul(flat_a, b)
            true_result = tf.reshape(flat_result, [8, 32, 32, 16])

            tf.debugging.assert_equal(true_result, einsum_result)

    def test_pca_constant(self):
        foo = tf.keras.Sequential([model.layers.PCA(2)])
        out = foo(tf.random.normal([32, 16, 16, 4]))
        tf.debugging.assert_shapes([(out, [32, 16, 16, 6])])
        self.assertEqual(len(foo.trainable_weights), 0)
        foo.trainable = True
        self.assertEqual(len(foo.trainable_weights), 0)

    def test_standardize(self):
        foo = model.layers.Standardize()

        x = tf.random.uniform([8, 32, 32, 3], maxval=255, dtype=tf.float32)
        y = foo(x)
        y_mean = tf.math.reduce_mean(y, axis=[0, 1, 2])
        y_var = tf.math.reduce_variance(y, axis=[0, 1, 2])
        tf.debugging.assert_near(y_mean, tf.zeros_like(y_mean), atol=1e-5, message='mean not zero')
        tf.debugging.assert_near(y_var, tf.ones_like(y_var), rtol=1e-5, message='variance not one')

        # Run the layer through a different distribution to make sure it doesn't affect the configured mean and variance
        x2 = tf.random.uniform([8, 32, 32, 3], minval=-255, maxval=-128, dtype=tf.float32)
        foo(x2)

        y = foo(x)
        y_mean = tf.math.reduce_mean(y, axis=[0, 1, 2])
        y_var = tf.math.reduce_variance(y, axis=[0, 1, 2])
        tf.debugging.assert_near(y_mean, tf.zeros_like(y_mean), atol=1e-5, message='mean not zero')
        tf.debugging.assert_near(y_var, tf.ones_like(y_var), rtol=1e-5, message='variance not one')

    def test_model_warmup(self):
        FLAGS(['', '--feat_model=fast'])
        feat_model = scm.make_feat_model([32, 32, 3])
        sc_model = scm.SCModel(feat_model, sample_size=None, loss_warmup=100)
        sc_model.compile('adam',
                         loss={'style': [tf.keras.losses.MeanSquaredError(), tf.keras.losses.MeanSquaredError()]})

        # Initial alpha value should be 0
        alpha = sc_model.get_loss_warmup_alpha()
        tf.debugging.assert_equal(tf.zeros_like(alpha), alpha)

        # Linear warmup to 1
        x = tf.random.uniform([1, 32, 32, 3], maxval=255, dtype=tf.int32)
        y = tf.random.uniform([1, 32, 32, 3], maxval=255, dtype=tf.int32)
        feats = {'style': [tf.random.uniform([1, 16, 16, 3]), tf.random.uniform([1, 8, 8, 3])],
                 'content': [tf.random.uniform([1, 16, 16, 3]), tf.random.uniform([1, 8, 8, 3])]}
        _ = sc_model.train_step(((x, y), feats))

        alpha = sc_model.get_loss_warmup_alpha()
        tf.debugging.assert_equal(0.01 * tf.ones_like(alpha), alpha)

        # Max value as 1
        for _ in range(200):
            _ = sc_model.train_step(((x, y), feats))

        alpha = sc_model.get_loss_warmup_alpha()
        tf.debugging.assert_equal(tf.ones_like(alpha), alpha)

    def test_model_no_warmup(self):
        FLAGS(['', '--feat_model=fast'])
        feat_model = scm.make_feat_model([32, 32, 3])
        sc_model = scm.SCModel(feat_model, sample_size=None, loss_warmup=0)
        sc_model.compile('adam',
                         loss={'style': [tf.keras.losses.MeanSquaredError(), tf.keras.losses.MeanSquaredError()]})

        # Initial alpha value should be 0
        alpha = sc_model.get_loss_warmup_alpha()
        tf.debugging.assert_equal(tf.ones_like(alpha), alpha)


if __name__ == '__main__':
    absltest.main()
