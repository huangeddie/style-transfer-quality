import os

import tensorflow as tf
from absl import flags, logging
from matplotlib import pyplot as plt
from tensorflow.keras import mixed_precision

FLAGS = flags.FLAGS

flags.DEFINE_string('style_image', None, 'path to the style image')
flags.DEFINE_string('content_image', None, 'path to the content image')
flags.DEFINE_integer('imsize', None, 'image size')

flags.DEFINE_bool('tpu', False, 'whether or not to use a tpu')
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

    content_image = style_image
    if FLAGS.content_image is not None:
        content_image = tf.image.decode_image(tf.io.read_file(FLAGS.content_image))
        if FLAGS.imsize is not None:
            content_image = tf.keras.preprocessing.image.smart_resize(content_image, [FLAGS.imsize, FLAGS.imsize])
        content_image = tf.image.convert_image_dtype(content_image, tf.float32)
        content_image = tf.expand_dims(content_image, 0)

    return style_image, content_image


def compute_skewness(x, axes):
    mu, var = tf.nn.moments(x, axes=axes, keepdims=True)

    z = (x - mu) * tf.math.rsqrt(var + 1e-3)

    skew = tf.reduce_mean(z ** 3, axis=axes, keepdims=True)
    return skew


def plot_metrics(logs_df, filename):
    logs_df.set_index('epoch')
    f, axes = plt.subplots(1, 4)
    f.set_size_inches(16, 3)
    logs_df.plot(y='loss', logy=True, ax=axes[0])
    logs_df.filter(like='mean').plot(logy=True, ax=axes[1])
    logs_df.filter(like='var').plot(logy=True, ax=axes[2])
    logs_df.filter(like='skew').plot(logy=True, ax=axes[3])
    f.tight_layout()
    f.savefig(os.path.join('./out', filename))


def log_metrics(logs_df):
    last_epoch_logs = logs_df.iloc[-1]
    logging.info(f"total loss: {last_epoch_logs['loss']:.4}")
    for metric in ['mean', 'var', 'skew']:
        mean_val = last_epoch_logs.filter(like=metric).sum()
        logging.info(f'total {metric:.4} loss: {mean_val:.4}')


def log_feat_distribution(feats_dict):
    moments = []
    for style_feats in feats_dict['style']:
        m1 = tf.reduce_mean(style_feats, axis=[1, 2]).numpy()
        m2 = tf.math.reduce_variance(style_feats, axis=[1, 2]).numpy()
        m3 = compute_skewness(style_feats, axes=[1, 2]).numpy()
        moments.append([m1, m2, m3])
    logging.info('=' * 100)
    logging.info('average style moments')
    logging.info(f"\tmean: {[m[0].mean() for m in moments]}")
    logging.info(f"\tvar: {[m[1].mean() for m in moments]}")
    logging.info(f"\tskew: {[m[2].mean() for m in moments]}")
    logging.info('=' * 100)
