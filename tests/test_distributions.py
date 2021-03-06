import tensorflow as tf
from absl import flags
from absl.testing import absltest
from scipy import stats

from distributions import compute_wass_dist, compute_raw_m2_loss

FLAGS = flags.FLAGS


class TestDistributions(absltest.TestCase):
    def test_wass_dist_shape(self):
        x = tf.random.normal([2, 32, 32, 8])
        y = tf.random.normal([2, 32, 32, 8])

        our_wass_dist = compute_wass_dist(y, x)
        tf.debugging.assert_shapes([(our_wass_dist, (2, 8))])

    def test_wass_dist(self):
        for _ in range(100):
            x = tf.random.normal([2, 32, 32, 8])
            y = tf.random.normal([2, 32, 32, 8])

            true_batch_wass_dist = []
            for i in range(2):
                true_feat_wass_dist = []
                for j in range(8):
                    true_feat_wass_dist.append(
                        stats.wasserstein_distance(x[i, :, :, j].numpy().flatten(), y[i, :, :, j].numpy().flatten()))
                true_batch_wass_dist.append(true_feat_wass_dist)
            true_batch_wass_dist = tf.constant(true_batch_wass_dist, dtype=tf.float32)

            our_wass_dist = compute_wass_dist(y, x)
            tf.debugging.assert_near(true_batch_wass_dist, our_wass_dist)

    def test_raw_m2_loss_shape(self):
        x = tf.random.normal([2, 32, 32, 8])
        y = tf.random.normal([2, 32, 32, 8])
        raw_m2_loss = compute_raw_m2_loss(x, y)
        tf.debugging.assert_shapes([(raw_m2_loss, (2, 8))])


if __name__ == '__main__':
    absltest.main()
