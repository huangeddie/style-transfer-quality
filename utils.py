import tensorflow as tf
from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_string("style_image", None, "path to the style image")
flags.DEFINE_string("content_image", None, "path to the content image")
flags.DEFINE_integer("imsize", None, "image size")

# Required flag.
flags.mark_flag_as_required("style_image")


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
