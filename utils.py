import os
import shutil

import tensorflow as tf
from absl import flags, logging
from matplotlib import pyplot as plt
from tensorflow.keras import mixed_precision

FLAGS = flags.FLAGS

flags.DEFINE_string('style_image', None, 'path to the style image')
flags.DEFINE_string('content_image', None, 'path to the content image')
flags.DEFINE_integer('imsize', None, 'image size')

flags.DEFINE_enum('strategy', None, ['tpu', 'multi_cpu'], 'distributed strategy')
flags.DEFINE_enum('policy', 'float32', ['float32', 'mixed_bfloat16'], 'floating point precision policy')


def setup():
    # Make base dir
    loss_dir = f'out/{FLAGS.loss}'
    shutil.rmtree(loss_dir, ignore_errors=True)
    os.mkdir(loss_dir)

    if FLAGS.strategy == 'tpu':
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)
        strategy = tf.distribute.TPUStrategy(resolver)
    elif FLAGS.strategy == 'multi_cpu':
        strategy = tf.distribute.MirroredStrategy(['CPU:0', 'CPU:1'])
    else:
        strategy = tf.distribute.get_strategy()

    # Policy
    policy = mixed_precision.Policy(FLAGS.policy)
    mixed_precision.set_global_policy(policy)

    return strategy, loss_dir


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


def get_layer_grams(layer_feats):
    grams = []
    for feats in layer_feats:
        num_locs = tf.cast(tf.reduce_prod(feats.shape[:-1]), tf.float32)
        grams.append(tf.einsum('bhwc,bhwd->bcd', feats, feats) / num_locs)
    return grams


def plot_loss(logs_df, path):
    logs_df = logs_df.filter(regex='^((?!epoch).)*$')
    nrows = len(logs_df.columns)
    f, axes = plt.subplots(nrows)
    f.set_size_inches(5, nrows * 5)
    for col, ax in zip(logs_df.columns, axes):
        logs_df[col].plot(logy=min(logs_df[col]) > 0, ax=ax)
        ax.set_title(col)
    f.tight_layout()
    f.savefig(path)


def log_feat_distribution(feats_dict, title):
    moments = []
    for style_feats in feats_dict['style']:
        m1 = tf.reduce_mean(style_feats, axis=[1, 2]).numpy()
        m2 = tf.math.reduce_variance(style_feats, axis=[1, 2]).numpy()
        m3 = compute_skewness(style_feats, axes=[1, 2]).numpy()
        moments.append([m1, m2, m3])
    logging.info('=' * 100)
    logging.info(title)
    logging.info(f"\tmean: {[m[0].mean() for m in moments]}")
    logging.info(f"\tvar: {[m[1].mean() for m in moments]}")
    logging.info(f"\tskew: {[m[2].mean() for m in moments]}")
    logging.info('=' * 100)


def plot_layer_grams(raw_feats_dict, feats_dict, filepath):
    raw_grams = get_layer_grams(raw_feats_dict['style'])
    proj_grams = get_layer_grams(feats_dict['style'])
    f, ax = plt.subplots(2, len(raw_grams), squeeze=False)
    f.set_size_inches(len(raw_grams) * 4, 5)
    for i, (raw_gram, proj_gram) in enumerate(zip(raw_grams, proj_grams)):
        ax[0, i].set_title(f'raw gram {i}')
        im = ax[0, i].imshow(tf.squeeze(raw_gram, 0))
        plt.colorbar(im, ax=ax[0, i])

        ax[1, i].set_title(f'proj gram {i}')
        im = ax[1, i].imshow(tf.squeeze(proj_gram, 0))
        plt.colorbar(im, ax=ax[1, i])
    f.tight_layout()
    f.savefig(filepath)
