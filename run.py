import pandas as pd
import tensorflow as tf
from absl import app
from absl import flags
from absl import logging

import model as scm
from training import train, compile_sc_model, make_dataset
from utils import plot_loss, log_feat_distribution, plot_layer_grams, setup, load_sc_images

FLAGS = flags.FLAGS

flags.DEFINE_enum('loss', None, ['m1', 'm1_m2', 'm1_covar', 'corawm2', 'wass'],
                  'type of statistical loss to use (optional)')
flags.DEFINE_integer('loss_warmup', 0, 'linear loss warmup')
flags.DEFINE_integer('sample_size', None, 'mini-batch sample size of the features per layer. '
                                          'defaults to using all the features per layer. '
                                          'if low on memory or want to speed up training, '
                                          'set this value to something like 1024')

flags.DEFINE_bool('train_metrics', True, 'measure metrics during training')


def main(argv):
    del argv  # Unused.

    # Setup
    strategy, loss_dir = setup()

    # Load style/content image
    logging.info('loading images')
    style_image, content_image = load_sc_images()

    # Create the style-content model
    logging.info('making style-content model')
    image_shape = style_image.shape[1:]
    with strategy.scope():
        raw_feat_model = scm.make_feat_model(image_shape)
        sc_model = scm.SCModel(raw_feat_model, FLAGS.sample_size, FLAGS.loss_warmup)

        # Configure the model to the style and content images
        sc_model.configure(style_image, content_image)

    # Plot the feature model structure
    tf.keras.utils.plot_model(sc_model.feat_model, './out/feat_model.jpg')

    # Get the style and content features
    raw_feats_dict = raw_feat_model((style_image, content_image), training=False)
    feats_dict = sc_model.feat_model((style_image, content_image), training=False)

    # Make the dataset
    ds = make_dataset(strategy, (style_image, content_image), feats_dict)

    # Log distribution statistics of the style image
    log_feat_distribution(raw_feats_dict, 'raw layer average style moments')
    log_feat_distribution(feats_dict, 'projected layer average style moments')

    # Plot the gram matrices
    plot_layer_grams(raw_feats_dict, feats_dict, filepath='./out/gram.jpg')

    # Reset gen image and recompile
    sc_model.reinit_gen_image()
    compile_sc_model(strategy, sc_model, FLAGS.loss, with_metrics=FLAGS.train_metrics)

    # Style transfer
    logging.info(f'loss function: {FLAGS.loss}')
    train(sc_model, ds, loss_dir)

    # Save the images to disk
    gen_image = sc_model.get_gen_image()
    for filename, image in [('style.jpg', style_image), ('content.jpg', content_image),
                            (f'{FLAGS.loss}.jpg', gen_image)]:
        tf.keras.preprocessing.image.save_img(f'{loss_dir}/{filename}', tf.squeeze(image, 0))
    logging.info(f'images saved to {loss_dir}')

    # Sanity evaluation
    logging.info('evaluating on projected features')
    compile_sc_model(strategy, sc_model, FLAGS.loss, with_metrics=True)
    sc_model.evaluate(ds, steps=1, return_dict=True)

    # Metrics
    logs_df = pd.read_csv(f'{loss_dir}/logs.csv')

    logging.info('evaluating on raw features')
    orig_feat_model = sc_model.feat_model
    sc_model.feat_model = raw_feat_model
    compile_sc_model(strategy, sc_model, FLAGS.loss, with_metrics=True)
    all_raw_metrics = sc_model.evaluate(ds, steps=1, return_dict=True)
    all_raw_metrics = pd.Series(all_raw_metrics)
    for metric in ['_mean', '_var', '_covar', '_gram', '_skew', '_wass']:
        raw_metrics = all_raw_metrics.filter(like=metric)
        raw_metrics[f'total{metric}_loss'] = raw_metrics.sum()
        filepath = f'{loss_dir}/raw_metrics.csv'
        raw_metrics.to_csv(filepath, mode='a', header=False)
        with open(filepath, mode='a') as f:
            f.write('\n')
    sc_model.feat_model = orig_feat_model

    plot_loss(logs_df, path=f'{loss_dir}/plots.jpg')
    logging.info(f'metrics saved to {loss_dir}')


if __name__ == '__main__':
    app.run(main)
