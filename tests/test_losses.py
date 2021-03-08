import tensorflow as tf
from absl import flags
from absl.testing import absltest

from distributions.losses import CoWassLoss, RandPairWassLoss

FLAGS = flags.FLAGS


class TestLosses(absltest.TestCase):
    def test_cowass_warmup(self):
        cowass = CoWassLoss()
        cowass.warmup_steps.assign(100)

        # Initial alpha value should be 0
        alpha = cowass.get_alpha()
        tf.debugging.assert_equal(tf.zeros_like(alpha), alpha)

        # Linear warmup to 1
        x = tf.random.normal([2, 32, 32, 8])
        y = tf.random.normal([2, 32, 32, 8])
        _ = cowass(x, y)

        alpha = cowass.get_alpha()
        tf.debugging.assert_equal(0.01 * tf.ones_like(alpha), alpha)

        # Max value as 1
        for _ in range(200):
            _ = cowass(x, y)

        alpha = cowass.get_alpha()
        tf.debugging.assert_equal(tf.ones_like(alpha), alpha)

    def test_cowass_no_warmup(self):
        cowass = CoWassLoss()

        # Initial alpha value should be 0
        alpha = cowass.get_alpha()
        tf.debugging.assert_equal(tf.ones_like(alpha), alpha)

    def test_rand_pair_wass(self):
        rp_wass = RandPairWassLoss()

        x = tf.random.normal([2, 32, 32, 8])
        y = tf.random.normal([2, 32, 32, 8])
        _ = rp_wass(x, y)


if __name__ == '__main__':
    absltest.main()
