import tensorflow as tf
from absl import flags
from absl.testing import absltest
from scipy import stats

from distributions import compute_wass_dist

FLAGS = flags.FLAGS


class TestDistributions(absltest.TestCase):
    def test_wass_dist(self):
        for _ in range(100):
            x = tf.random.normal([1, 32, 32, 8])
            y = tf.random.normal([1, 32, 32, 8])

            true_wass_dist = []
            for i in range(8):
                true_wass_dist.append(
                    stats.wasserstein_distance(x[:, :, :, i].numpy().flatten(), y[:, :, :, i].numpy().flatten()))
            true_wass_dist = tf.constant([true_wass_dist], dtype=tf.float32)

            our_wass_dist = compute_wass_dist(y, x)
            tf.debugging.assert_near(true_wass_dist, our_wass_dist)


if __name__ == '__main__':
    absltest.main()
