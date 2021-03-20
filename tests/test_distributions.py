import tensorflow as tf
from absl import flags
from absl.testing import absltest
from scipy import stats

from distributions import compute_wass_dist, compute_raw_m2_loss, compute_mean_loss, compute_var_loss, \
    compute_covar_loss, compute_skew_loss, sample_k

FLAGS = flags.FLAGS


class TestDistributions(absltest.TestCase):
    def test_fn_signature(self):
        x = tf.random.normal([2, 1024, 8])
        y = tf.random.normal([2, 1024, 8])

        for fn in [compute_wass_dist, compute_raw_m2_loss, compute_mean_loss, compute_var_loss,
                   compute_covar_loss, compute_skew_loss]:
            z = fn(x, y, p=1)
            tf.debugging.assert_shapes([(z, [2])], message=str(fn))

    def test_wass_dist(self):
        for _ in range(100):
            x = tf.random.normal([2, 1024, 8])
            y = tf.random.normal([2, 1024, 8])

            true_batch_wass_dist = []
            for i in range(2):
                true_feat_wass_dist = []
                for j in range(8):
                    true_feat_wass_dist.append(
                        stats.wasserstein_distance(x[i, :, j].numpy().flatten(), y[i, :, j].numpy().flatten()))
                true_batch_wass_dist.append(tf.reduce_mean(true_feat_wass_dist))
            true_batch_wass_dist = tf.concat(true_batch_wass_dist, axis=0)
            true_batch_wass_dist = tf.cast(true_batch_wass_dist, tf.float32)

            our_wass_dist = compute_wass_dist(y, x, p=1)
            tf.debugging.assert_near(true_batch_wass_dist, our_wass_dist)

    def test_sampling(self):
        x = tf.random.normal([2, 1024, 8])
        sample_x1 = sample_k(x, None)
        sample_x2 = sample_k(x, 256)
        sample_x3 = sample_k(x, 2048)
        tf.debugging.assert_shapes([
            (sample_x1, [2, 1024, 8]),
            (sample_x2, [2, 256, 8]),
            (sample_x3, [2, 1024, 8]),
        ])


if __name__ == '__main__':
    absltest.main()
