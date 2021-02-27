import tensorflow as tf
from absl import flags
from absl.testing import absltest

import utils

FLAGS = flags.FLAGS


class TestUtils(absltest.TestCase):
    def test_load_sc_images(self):
        FLAGS(['', '--style_image=../imgs/starry_night.jpg'])
        style_image, content_image = utils.load_sc_images()

        tf.debugging.assert_rank(style_image, 4)
        tf.debugging.assert_rank(content_image, 4)

        tf.debugging.assert_type(style_image, tf.float32)
        tf.debugging.assert_type(content_image, tf.float32)

        tf.debugging.assert_greater_equal(style_image, tf.zeros_like(style_image))
        tf.debugging.assert_less_equal(style_image, tf.ones_like(style_image))

        if tf.reduce_all(tf.math.is_finite(content_image)):
            tf.debugging.assert_greater_equal(content_image, tf.zeros_like(content_image))
            tf.debugging.assert_less_equal(content_image, tf.ones_like(content_image))


if __name__ == '__main__':
    absltest.main()
