import datetime
import os

import pandas as pd
import tensorflow as tf
from absl import app
from absl import flags
from absl import logging

import style_content as sc
import utils
from utils import plot_metrics, log_metrics, log_feat_distribution

FLAGS = flags.FLAGS

flags.DEFINE_multi_enum('losses', ['m2'], ['m2', 'gram', 'm3'], 'type of loss to use')

flags.DEFINE_float('lr', 1e-3, 'learning rate')
flags.DEFINE_float('beta1', 0.9, 'beta1')
flags.DEFINE_float('beta2', 0.99, 'beta2')
flags.DEFINE_float('epsilon', 1e-5, 'epsilon')

flags.DEFINE_integer('train_steps', 100, 'train steps')

flags.DEFINE_bool('load', False, 'load')


def run_style_transfer(strategy, style_image, content_image, loss_key):
    # Create the style-content model
    logging.info('making style-content model')
    image_shape = style_image.shape[1:]
    sc_model = sc.make_sc_model(strategy, image_shape, loss_key)

    # Plot the feature model structure
    tf.keras.utils.plot_model(sc_model.feat_model, './out/feat_model.jpg')

    # Configure batch norm layers to normalize features of the style and content images
    sc_model.feat_model((style_image, content_image), training=True)
    sc_model.feat_model.trainable = False

    # Get the style and content features
    feats_dict = sc_model.feat_model((style_image, content_image), training=False)

    # Log distribution statistics of the style image
    log_feat_distribution(feats_dict)

    # Run the style model
    logging.info(f'loss function: {loss_key}')
    start_time = datetime.datetime.now()
    sc_model.fit((style_image, content_image), feats_dict, epochs=FLAGS.train_steps, batch_size=1,
                 verbose=FLAGS.verbose, callbacks=tf.keras.callbacks.CSVLogger('./out/logs.csv', append=FLAGS.load))
    end_time = datetime.datetime.now()
    duration = end_time - start_time
    logging.info(f'training took {duration}')

    # Sanity evaluation
    sc_model.evaluate((style_image, content_image), feats_dict, batch_size=1, return_dict=True)

    # Save the images to disk
    gen_image = sc_model.get_gen_image()
    for filename, image in [('style.jpg', style_image), ('content.jpg', content_image), (f'{loss_key}.jpg', gen_image)]:
        tf.keras.preprocessing.image.save_img(os.path.join('./out', filename), tf.squeeze(image, 0))
    logging.info(f'images saved to ./out')

    # Metrics
    logs_df = pd.read_csv('out/logs.csv')

    # Print contributing loss of each metric
    log_metrics(logs_df)

    # Plot metrics
    plot_metrics(logs_df)


def main(argv):
    del argv  # Unused.
    logging.info(FLAGS.losses)
    strategy = utils.setup()

    # Load style/content image
    logging.info('loading images')
    style_image, content_image = utils.load_sc_images()

    for loss_key in FLAGS.losses:
        run_style_transfer(strategy, style_image, content_image, loss_key)


if __name__ == '__main__':
    app.run(main)
