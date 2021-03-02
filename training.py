import datetime

import tensorflow as tf
from absl import flags
from absl import logging

import dist_losses
import dist_metrics

FLAGS = flags.FLAGS

flags.DEFINE_integer('train_steps', 100, 'train steps')
flags.DEFINE_integer('verbose', 0, 'verbosity')


def train(sc_model, style_image, content_image, feats_dict, loss_key):
    start_time = datetime.datetime.now()
    sc_model.fit((style_image, content_image), feats_dict, epochs=FLAGS.train_steps, batch_size=1,
                 verbose=FLAGS.verbose, callbacks=tf.keras.callbacks.CSVLogger(f'./out/{loss_key}_logs.csv'))
    end_time = datetime.datetime.now()
    duration = end_time - start_time
    logging.info(f'training took {duration}')


def compile_sc_model(strategy, sc_model, loss_key):
    with strategy.scope():
        loss_dict = {'style': [dist_losses.loss_dict[loss_key] for _ in sc_model.feat_model.output['style']]}
        metrics = [dist_metrics.MeanLoss(), dist_metrics.VarLoss(), dist_metrics.GramLoss(), dist_metrics.SkewLoss()]
        metric_dict = {'style': [metrics for _ in sc_model.feat_model.output['style']],
                       'content': [[] for _ in sc_model.feat_model.output['content']]}
        if FLAGS.content_image is not None:
            loss_dict['content'] = [tf.keras.losses.MeanSquaredError() for _ in sc_model.feat_model.output['content']]

        sc_model.compile(tf.keras.optimizers.Adam(FLAGS.lr, FLAGS.beta1, FLAGS.beta2, FLAGS.epsilon),
                         loss=loss_dict, metrics=metric_dict)
    return sc_model
