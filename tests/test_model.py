import tensorflow as tf
from absl import flags
from absl.testing import absltest

import style_content as sc

FLAGS = flags.FLAGS


class TestModel(absltest.TestCase):
    def test_model_train_step(self):
        FLAGS(['', '--feat_model=fast'])
        sc_model = sc.SCModel([32, 32, 3])
        # Random uniform doesn't support uint8
        x = tf.random.uniform([1, 32, 32, 3], maxval=255, dtype=tf.int32)
        y = tf.random.uniform([1, 32, 32, 3], maxval=255, dtype=tf.int32)
        metrics = sc_model.train_step((x, y))
        self.assertIsInstance(metrics, dict)

    def test_model_call(self):
        FLAGS(['', '--feat_model=fast'])
        sc_model = sc.SCModel([32, 32, 3])
        # Random uniform doesn't support uint8
        x = tf.random.uniform([1, 32, 32, 3], maxval=255, dtype=tf.int32)
        y = tf.random.uniform([1, 32, 32, 3], maxval=255, dtype=tf.int32)
        output = sc_model((x, y))
        print(output)


if __name__ == '__main__':
    absltest.main()
