import pandas as pd
import tensorflow as tf
from absl import app
from absl import flags
from absl import logging

import style_content_model as scm
import utils
from training import train, compile_sc_model
from utils import plot_metrics, log_feat_distribution

FLAGS = flags.FLAGS

flags.DEFINE_multi_enum('losses', ['m2'], ['m1', 'm2', 'gram', 'm3'], 'type of loss to use')


def main(argv):
    del argv  # Unused.

    # Setup
    strategy = utils.setup()

    # Load style/content image
    logging.info('loading images')
    style_image, content_image = utils.load_sc_images()

    # Create the style-content model
    logging.info('making style-content model')
    image_shape = style_image.shape[1:]
    with strategy.scope():
        raw_feat_model = scm.make_feat_model(image_shape)
        sc_model = scm.SCModel(raw_feat_model)
        scm.configure(sc_model, style_image, content_image)

    # Plot the feature model structure
    tf.keras.utils.plot_model(sc_model.feat_model, './out/feat_model.jpg')

    # Get the style and content features
    raw_feat_dict = raw_feat_model((style_image, content_image), training=False)
    feats_dict = sc_model.feat_model((style_image, content_image), training=False)

    # Log distribution statistics of the style image
    log_feat_distribution(feats_dict)

    # Run the transfer for each loss
    for loss_key in FLAGS.losses:
        # Reset gen image and recompile
        sc_model.reinit_gen_image()
        sc_model = compile_sc_model(strategy, sc_model, loss_key)

        # Style transfer
        logging.info(f'loss function: {loss_key}')
        train(sc_model, style_image, content_image, feats_dict, loss_key)

        # Sanity evaluation
        sc_model.evaluate((style_image, content_image), feats_dict, batch_size=1, return_dict=True)

        # Save the images to disk
        gen_image = sc_model.get_gen_image()
        for filename, image in [('style.jpg', style_image), ('content.jpg', content_image),
                                (f'{loss_key}_gen.jpg', gen_image)]:
            tf.keras.preprocessing.image.save_img(f'./out/{filename}', tf.squeeze(image, 0))
        logging.info('images saved to ./out')

        # Metrics
        logs_df = pd.read_csv(f'out/{loss_key}_logs.csv')

        orig_feat_model = sc_model.feat_model
        sc_model.feat_model = raw_feat_model
        sc_model = compile_sc_model(strategy, sc_model, loss_key)
        raw_metrics = sc_model.evaluate((style_image, content_image), raw_feat_dict, batch_size=1, return_dict=True)
        raw_metrics = pd.Series(raw_metrics)
        raw_metrics.to_csv(f'out/{loss_key}_raw_metrics.csv')
        sc_model.feat_model = orig_feat_model

        plot_metrics(logs_df, filename=f'{loss_key}_plots.jpg')
        logging.info('metrics saved to ./out')


if __name__ == '__main__':
    app.run(main)
