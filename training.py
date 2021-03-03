import datetime

import tensorflow as tf
from absl import flags
from absl import logging

from distributions import losses, metrics

FLAGS = flags.FLAGS

flags.DEFINE_integer('train_steps', 100, 'train steps')
flags.DEFINE_integer('verbose', 0, 'verbosity')


def train(sc_model, style_image, content_image, feats_dict, callbacks):
    start_time = datetime.datetime.now()
    sc_model.fit((style_image, content_image), feats_dict, epochs=FLAGS.train_steps, batch_size=1,
                 verbose=FLAGS.verbose, callbacks=callbacks)
    end_time = datetime.datetime.now()
    duration = end_time - start_time
    logging.info(f'training took {duration}')


def compile_sc_model(strategy, sc_model, loss_key):
    with strategy.scope():
        loss_dict = {'style': [losses.loss_dict[loss_key] for _ in sc_model.feat_model.output['style']]}
        metric_dict = {'style': [[metrics.WassDist(), metrics.MeanLoss(), metrics.VarLoss(), metrics.GramLoss(), metrics.SkewLoss()]
                                 for _ in sc_model.feat_model.output['style']],
                       'content': [[] for _ in sc_model.feat_model.output['content']]}
        if FLAGS.content_image is not None:
            loss_dict['content'] = [tf.keras.losses.MeanSquaredError() for _ in sc_model.feat_model.output['content']]

        lr_schedule = tf.keras.optimizers.schedules.CosineDecay(FLAGS.lr, FLAGS.train_steps)
        sc_model.compile(tf.keras.optimizers.Adam(lr_schedule, FLAGS.beta1, FLAGS.beta2, FLAGS.epsilon),
                         loss=loss_dict, metrics=metric_dict)
    return sc_model
