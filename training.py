import datetime

import tensorflow as tf
import tensorflow_addons as tfa
from absl import flags
from absl import logging

from distributions import losses, metrics

FLAGS = flags.FLAGS

flags.DEFINE_integer('train_steps', 100, 'train steps')
flags.DEFINE_integer('steps_exec', 1, 'steps per execution')
flags.DEFINE_integer('sample_size', None, 'sample size of the features per layer')
flags.DEFINE_integer('cowass_warmup', 0, 'warmup steps for the CoWass loss')
flags.DEFINE_integer('verbose', 0, 'verbosity')
flags.DEFINE_bool('cosine_decay', False, 'cosine decay')

flags.DEFINE_enum('opt', 'lamb', ['adam', 'lamb'], 'optimizer')
flags.DEFINE_float('lr', 1e-3, 'learning rate')
flags.DEFINE_float('beta1', 0.9, 'beta1')
flags.DEFINE_float('beta2', 0.99, 'beta2')
flags.DEFINE_float('epsilon', 1e-7, 'epsilon')


def make_dataset(strategy, images, feats_dict):
    images_ds = tf.data.Dataset.from_tensor_slices(images)
    style_feats = [tf.data.Dataset.from_tensor_slices(feats) for feats in feats_dict['style']]
    style_feats = tuple(style_feats)
    style_ds = tf.data.Dataset.zip(style_feats)

    content_feats = [tf.data.Dataset.from_tensor_slices(feats) for feats in feats_dict['content']]
    content_feats = tuple(content_feats)
    content_ds = tf.data.Dataset.zip(content_feats)

    feats_ds = tf.data.Dataset.zip((style_ds, content_ds))
    feats_ds = feats_ds.map(lambda x, y: {'style': x, 'content': y})
    ds = tf.data.Dataset.zip((images_ds, feats_ds))
    ds = ds.cache().repeat().batch(strategy.num_replicas_in_sync, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
    logging.info(f'dataset: {ds}')
    dist_ds = strategy.experimental_distribute_dataset(ds)
    return dist_ds


def train(sc_model, ds, callbacks):
    start_time = datetime.datetime.now()
    try:
        history = sc_model.fit(ds, epochs=FLAGS.train_steps // FLAGS.steps_exec,
                               steps_per_epoch=FLAGS.steps_exec, verbose=FLAGS.verbose, callbacks=callbacks)
        for key, val in history.history.items():
            history.history[key] = val[-1]
        logging.info(history.history)
    except KeyboardInterrupt:
        logging.info('caught keyboard interrupt. ended training early')
    end_time = datetime.datetime.now()
    duration = end_time - start_time
    logging.info(f'training took {duration}')


def compile_sc_model(strategy, sc_model, loss_key, with_metrics):
    with strategy.scope():
        # Style loss
        loss_dict = {
            'style': [losses.loss_dict[loss_key](FLAGS.sample_size) for _ in sc_model.feat_model.output['style']]
        }

        # Content loss
        if FLAGS.content_image is not None:
            loss_dict['content'] = [tf.keras.losses.MeanSquaredError() for _ in sc_model.feat_model.output['content']]

        # Configure the CoWass loss if any
        for loss_list in loss_dict.values():
            for loss in loss_list:
                if isinstance(loss, losses.CoWassLoss):
                    loss.warmup_steps.assign(FLAGS.cowass_warmup)

        # Learning rate schedule
        if FLAGS.cosine_decay:
            lr_schedule = tf.keras.experimental.CosineDecay(FLAGS.lr, FLAGS.train_steps)
            logging.info(f'using cosine decay lr schedule with lr={FLAGS.lr}, train_steps={FLAGS.train_steps}')
        else:
            lr_schedule = FLAGS.lr

        if FLAGS.opt == 'adam':
            optimizer = tf.keras.optimizers.Adam(lr_schedule, FLAGS.beta1, FLAGS.beta2, FLAGS.epsilon)
        elif FLAGS.opt == 'lamb':
            optimizer = tfa.optimizers.LAMB(lr_schedule, FLAGS.beta1, FLAGS.beta2, FLAGS.epsilon)
        else:
            raise ValueError(f'unknown optimizer: {FLAGS.opt}')
        logging.info(f'using the {optimizer.__class__.__name__} optimizer')

        # Metrics?
        if with_metrics:
            metric_dict = {'style': [
                [metrics.WassDist(), metrics.MeanLoss(), metrics.VarLoss(), metrics.CovarLoss(), metrics.SkewLoss(),
                 metrics.GramLoss()]
                for _ in sc_model.feat_model.output['style']],
                'content': [[] for _ in sc_model.feat_model.output['content']]}
        else:
            metric_dict = None

        # Compile
        sc_model.compile(optimizer, loss=loss_dict, metrics=metric_dict, steps_per_execution=FLAGS.steps_exec)
