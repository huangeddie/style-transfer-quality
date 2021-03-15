import datetime
import os
import shutil

import tensorflow as tf
import tensorflow_addons as tfa
from absl import flags
from absl import logging

from distributions import losses, metrics

FLAGS = flags.FLAGS

flags.DEFINE_integer('train_steps', 100, 'train steps')
flags.DEFINE_integer('steps_exec', 1, 'steps per execution')
flags.DEFINE_integer('cowass_warmup', 0, 'warmup steps for the CoWass loss')
flags.DEFINE_bool('cosine_decay', False, 'cosine decay')

flags.DEFINE_integer('verbose', 0, 'verbosity')
flags.DEFINE_bool('checkpoints', False, 'save transfer image every epoch')

flags.DEFINE_float('disc_lr', 1e-2, 'discriminator learning rate')
flags.DEFINE_float('gen_lr', 1, 'learning rate')
flags.DEFINE_float('beta1', 0.9, 'beta1')
flags.DEFINE_float('beta2', 0.99, 'beta2')
flags.DEFINE_float('epsilon', 1e-7, 'epsilon')


class TransferCheckpoint(tf.keras.callbacks.Callback):
    def __init__(self, out_dir, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.out_dir = out_dir
        self.checkpoint_dir = os.path.join(self.out_dir, 'checkpoints')
        if os.path.exists(self.checkpoint_dir):
            shutil.rmtree(self.checkpoint_dir)
            os.mkdir(self.checkpoint_dir)

    def save_transfer(self, iteration):
        image = tf.squeeze(self.model.gen_image, 0)
        encoded_image = tf.io.encode_jpeg(tf.cast(image, tf.uint8))
        tf.io.write_file(os.path.join(self.checkpoint_dir, f'{iteration:05d}.jpg'), encoded_image)

    def on_train_begin(self, logs=None):
        self.save_transfer(0)

    def on_epoch_end(self, epoch, logs=None):
        self.save_transfer(epoch)


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


def train(sc_model, ds, out_dir):
    start_time = datetime.datetime.now()
    try:
        callbacks = [
            tf.keras.callbacks.CSVLogger(f'{out_dir}/logs.csv'),
        ]
        if FLAGS.checkpoints:
            callbacks.append(TransferCheckpoint(out_dir))
            logging.info('saving checkpoints')

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
        loss_dict = {'style': [losses.loss_dict[loss_key]() for _ in sc_model.feat_model.output['style']]}

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
            disc_schedule = tf.keras.experimental.CosineDecay(FLAGS.disc_lr, FLAGS.train_steps)
            gen_schedule = tf.keras.experimental.CosineDecay(FLAGS.gen_lr, FLAGS.train_steps)
            logging.info(f'using cosine decay lr schedule')
        else:
            disc_schedule = FLAGS.disc_lr
            gen_schedule = FLAGS.gen_lr

        disc_opt = tfa.optimizers.LAMB(disc_schedule)
        gen_opt = tf.keras.optimizers.Adam(gen_schedule, FLAGS.beta1, FLAGS.beta2, FLAGS.epsilon)

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
        sc_model.compile(disc_opt, gen_opt, loss=loss_dict, metrics=metric_dict, steps_per_execution=FLAGS.steps_exec)
