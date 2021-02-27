import os

import tensorflow as tf
from absl import app
from absl import flags
from absl import logging

import style_content as sc
import utils

FLAGS = flags.FLAGS


def main(argv):
    del argv  # Unused.

    # Load style/content image
    logging.info('loading images')
    style_image, content_image = utils.load_sc_images()

    # Setup generated image
    gen_image = utils.setup_gen_image()

    # Create the style-content model
    logging.info('making style-content model')
    sc_model = sc.make_sc_model()

    # Run the style model
    logging.info('running style transfer')
    gen_image = sc_model.transfer(style_image, content_image, gen_image)
    logging.info('style transfer complete')

    # Save the generated image to disk
    gen_path = os.path.join('./out', 'gen.png')
    tf.keras.preprocessing.image.save_img(gen_path, gen_image, data_format='channels_last')
    logging.info(f'generated image saved to {gen_path}')


if __name__ == '__main__':
    app.run(main)
