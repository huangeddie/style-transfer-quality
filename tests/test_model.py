import tensorflow as tf
from absl import flags
from absl.testing import absltest

import style_content_model as scm

FLAGS = flags.FLAGS


class TestModel(absltest.TestCase):
    def test_model_train_step(self):
        FLAGS(['', '--feat_model=fast'])
        feat_model = scm.make_feat_model([32, 32, 3])
        sc_model = scm.SCModel(feat_model)
        # Random uniform doesn't support uint8
        x = tf.random.uniform([1, 32, 32, 3], maxval=255, dtype=tf.int32)
        y = tf.random.uniform([1, 32, 32, 3], maxval=255, dtype=tf.int32)
        metrics = sc_model.train_step((x, y))
        self.assertIsInstance(metrics, dict)

    def test_model_call(self):
        FLAGS(['', '--feat_model=fast'])
        feat_model = scm.make_feat_model([32, 32, 3])
        sc_model = scm.SCModel(feat_model)
        # Random uniform doesn't support uint8
        x = tf.random.uniform([1, 32, 32, 3], maxval=255, dtype=tf.int32)
        y = tf.random.uniform([1, 32, 32, 3], maxval=255, dtype=tf.int32)
        output = sc_model((x, y))
        print(output)

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
        foo = tf.keras.Sequential([scm.PCA(2)])
        out = foo(tf.random.normal([32, 16, 16, 4]))
        tf.debugging.assert_shapes([(out, [32, 16, 16, 2])])
        self.assertEqual(len(foo.trainable_weights), 0)
        foo.trainable = True
        self.assertEqual(len(foo.trainable_weights), 0)


if __name__ == '__main__':
    absltest.main()
