import os

import tensorflow as tf
from absl import app
from absl import flags
from absl import logging

import style_content as sc
import utils

FLAGS = flags.FLAGS
flags.DEFINE_float("lr", 1e-3, "learning rate")
flags.DEFINE_float("beta1", 0.5, "learning rate")
flags.DEFINE_float("beta2", 0.75, "learning rate")
flags.DEFINE_integer("train_steps", 1000, "train steps")


def main(argv):
    del argv  # Unused.

    # Load style/content image
    logging.info('loading images')
    style_image, content_image = utils.load_sc_images()

    # Create the style-content model
    logging.info('making style-content model')
    sc_model = sc.SCModel()
    if FLAGS.cache_feats:
        logging.info('caching style and content features')
        sc_model.cache_feats(style_image, content_image)
    sc_model.compile(tf.keras.optimizers.Adam(FLAGS.lr, FLAGS.beta1, FLAGS.beta2))

    # Run the style model
    sc_model.fit(style_image, content_image, steps_per_epoch=FLAGS.train_steps)

    # Make the generated image
    logging.info('generating style transfer image')
    gen_image = sc_model.get_gen_image()

    # Save the generated image to disk
    gen_path = os.path.join('./out', 'gen.png')
    tf.keras.preprocessing.image.save_img(gen_path, tf.squeeze(gen_image, 0), data_format='channels_last')
    logging.info(f'generated image saved to {gen_path}')


if __name__ == '__main__':
    app.run(main)
