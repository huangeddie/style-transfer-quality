import tensorflow as tf
from absl import flags
from tensorflow.keras import mixed_precision

FLAGS = flags.FLAGS

flags.DEFINE_string('style_image', None, 'path to the style image')
flags.DEFINE_string('content_image', None, 'path to the content image')
flags.DEFINE_integer('imsize', None, 'image size')

flags.DEFINE_bool('tpu', True, 'whether or not to use a tpu')
flags.DEFINE_enum('policy', 'float32', ['float32', 'mixed_bfloat16'], 'floating point precision policy')

# Required flag.
flags.mark_flag_as_required('style_image')


def setup():
    if FLAGS.tpu:
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)
        strategy = tf.distribute.TPUStrategy(resolver)
    else:
        strategy = tf.distribute.get_strategy()

    # Policy
    policy = mixed_precision.Policy(FLAGS.policy)
    mixed_precision.set_global_policy(policy)

    return strategy


def load_sc_images():
    style_image = tf.image.decode_image(tf.io.read_file(FLAGS.style_image))
    if FLAGS.imsize is not None:
        style_image = tf.keras.preprocessing.image.smart_resize(style_image, [FLAGS.imsize, FLAGS.imsize])
    style_image = tf.image.convert_image_dtype(style_image, tf.float32)
    style_image = tf.expand_dims(style_image, 0)

    content_image = tf.ones_like(style_image) * float('nan')
    if FLAGS.content_image is not None:
        content_image = tf.image.decode_image(tf.io.read_file(FLAGS.content_image))
        if FLAGS.imsize is not None:
            content_image = tf.keras.preprocessing.image.smart_resize(content_image, [FLAGS.imsize, FLAGS.imsize])
        content_image = tf.image.convert_image_dtype(content_image, tf.float32)
        content_image = tf.expand_dims(content_image, 0)

    return style_image, content_image
