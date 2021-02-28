import os

import tensorflow as tf
from absl import app
from absl import flags
from absl import logging

import discriminators as disc
import style_content as sc
import utils

FLAGS = flags.FLAGS
flags.DEFINE_float('lr', 1e-3, 'learning rate')
flags.DEFINE_float('beta1', 0.5, 'learning rate')
flags.DEFINE_float('beta2', 0.75, 'learning rate')
flags.DEFINE_integer('train_steps', 100, 'train steps')


def main(argv):
    del argv  # Unused.

    strategy = utils.setup()

    # Load style/content image
    logging.info('loading images')
    style_image, content_image = utils.load_sc_images()

    # Create the style-content model
    logging.info('making style-content model')
    with strategy.scope():
        sc_model = sc.SCModel(style_image.shape[1:])
    sc_model.compile(tf.keras.optimizers.Adam(FLAGS.lr, FLAGS.beta1, FLAGS.beta2),
                     loss={'style': disc.FirstMomentLoss(), 'content': tf.keras.losses.MeanSquaredError()})

    # Run the style model
    feats_dict = sc_model((style_image, content_image))
    sc_model.fit((style_image, content_image), feats_dict, epochs=FLAGS.train_steps, batch_size=1, verbose=0)

    # Get generated image
    gen_image = sc_model.get_gen_image()

    # Save the generated image to disk
    tf.keras.preprocessing.image.save_img(os.path.join('./out', 'style.png'), tf.squeeze(style_image, 0))
    tf.keras.preprocessing.image.save_img(os.path.join('./out', 'content.png'), tf.squeeze(content_image, 0))
    tf.keras.preprocessing.image.save_img(os.path.join('./out', 'gen.png'), tf.squeeze(gen_image, 0))
    logging.info(f'images saved to ./out')


if __name__ == '__main__':
    app.run(main)
