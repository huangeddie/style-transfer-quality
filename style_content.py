import tensorflow as tf
from absl import flags
from absl import logging
from sklearn import decomposition

import dist_losses
import dist_metrics

FLAGS = flags.FLAGS

flags.DEFINE_enum('feat_model', 'vgg19', ['vgg19', 'nasnetlarge', 'fast'],
                  'whether or not to cache the features when performing style transfer')
flags.DEFINE_integer('pca', None, 'maximum dimension of features enforced with PCA')

flags.DEFINE_float('lr', 1e-3, 'learning rate')
flags.DEFINE_float('beta1', 0.9, 'beta1')
flags.DEFINE_float('beta2', 0.99, 'beta2')
flags.DEFINE_float('epsilon', 1e-7, 'epsilon')


class Preprocess(tf.keras.layers.Layer):
    def __init__(self, preprocess_fn, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.preprocess = preprocess_fn

    def call(self, inputs, **kwargs):
        return self.preprocess(inputs)


class PCA(tf.keras.layers.Layer):
    def __init__(self, out_dim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.out_dim = out_dim

    def build(self, input_shape):
        self.projection = self.add_weight('projection', [input_shape[-1], self.out_dim], trainable=False)

    def configure(self, feats):
        pca = decomposition.PCA(n_components=self.out_dim)
        channels = tf.shape(feats)[-1]
        pca.fit(tf.reshape(feats, [-1, channels]))
        self.projection.assign(tf.constant(pca.components_.T, dtype=self.projection.dtype))

    def call(self, inputs, **kwargs):
        return tf.einsum('bhwc,cd->bhwd', inputs, self.projection)


class SCModel(tf.keras.Model):
    def __init__(self, feat_model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.feat_model = feat_model

    def build(self, input_shape):
        self.gen_image = self.add_weight('gen_image', input_shape[0],
                                         initializer=tf.keras.initializers.RandomUniform(minval=0, maxval=255))

    def reinit_gen_image(self):
        self.gen_image.assign(tf.random.uniform(self.gen_image.shape, maxval=255, dtype=self.gen_image.dtype))

    def call(self, inputs, training=None, mask=None):
        return self.feat_model((self.gen_image, self.gen_image), training=training)

    def train_step(self, data):
        images, feats = data

        with tf.GradientTape() as tape:
            # Compute generated features
            gen_feats = self(images, training=False)

            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = self.compiled_loss(feats, gen_feats, regularization_losses=self.losses)

        # Optimize generated image
        grad = tape.gradient(loss, [self.gen_image])
        self.optimizer.apply_gradients(zip(grad, [self.gen_image]))
        # Clip to RGB range
        self.gen_image.assign(tf.clip_by_value(self.gen_image, 0, 255))

        # Update metrics
        self.compiled_metrics.update_state(feats, gen_feats)

        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

    def get_gen_image(self):
        return tf.constant(tf.cast(self.gen_image, tf.uint8))


def make_feat_model(input_shape):
    style_input = tf.keras.Input(input_shape, name='style')
    content_input = tf.keras.Input(input_shape, name='content')
    if FLAGS.feat_model == 'vgg19':
        preprocess_fn = Preprocess(tf.keras.applications.vgg19.preprocess_input)
        vgg19 = tf.keras.applications.VGG19(include_top=False)
        vgg19.trainable = False

        content_layers = ['block5_conv2']
        style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
        vgg_style_outputs = [vgg19.get_layer(name).output for name in style_layers]
        vgg_content_outputs = [vgg19.get_layer(name).output for name in content_layers]

        vgg_style = tf.keras.Model(vgg19.input, vgg_style_outputs)
        vgg_content = tf.keras.Model(vgg19.input, vgg_content_outputs)

        x = preprocess_fn(style_input)
        style_output = vgg_style(x)

        x = preprocess_fn(content_input)
        content_output = vgg_content(x)

    elif FLAGS.feat_model == 'nasnetlarge':
        preprocess_fn = Preprocess(tf.keras.applications.nasnet.preprocess_input)
        nasnet = tf.keras.applications.NASNetLarge(include_top=False)
        nasnet.trainable = False

        content_layers = ['normal_conv_1_16']
        style_layers = ['normal_conv_1_0', 'normal_conv_1_4', 'normal_conv_1_8', 'normal_conv_1_12', 'normal_conv_1_16']
        nasnet_style_outputs = [nasnet.get_layer(name).output for name in style_layers]
        nasnet_content_outputs = [nasnet.get_layer(name).output for name in content_layers]

        nasnet_style = tf.keras.Model(nasnet.input, nasnet_style_outputs)
        nasnet_content = tf.keras.Model(nasnet.input, nasnet_content_outputs)

        x = preprocess_fn(style_input)
        style_output = nasnet_style(x)

        x = preprocess_fn(content_input)
        content_output = nasnet_content(x)

    elif FLAGS.feat_model == 'fast':
        avg_pool = tf.keras.layers.AveragePooling2D(pool_size=4)

        style_output = avg_pool(style_input)
        content_output = avg_pool(content_input)

    else:
        raise ValueError(f'unknown feature model: {FLAGS.feat_model}')

    style_model = tf.keras.Model(style_input, style_output)
    content_model = tf.keras.Model(content_input, content_output)
    new_style_outputs = [tf.keras.layers.BatchNormalization(scale=False, center=False, momentum=0)(output) for
                         output in style_model.outputs]
    new_content_outputs = [tf.keras.layers.BatchNormalization(scale=False, center=False, momentum=0)(output) for
                           output in content_model.outputs]

    return tf.keras.Model([style_model.input, content_model.input],
                          {'style': new_style_outputs, 'content': new_content_outputs})


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


def configure_sc_model(sc_model, style_image, content_image):
    feat_model = sc_model.feat_model

    # Build the gen image
    sc_model((style_image, content_image))

    # Configure the batch normalization layer
    feats_dict = feat_model((style_image, content_image), training=True)
    feat_model.trainable = False

    # Configure the PCA layers if any
    if FLAGS.pca is not None:
        new_style_outputs = []
        for old_output, feats, in zip(feat_model.output['style'], feats_dict['style']):
            pca = PCA(min(FLAGS.pca, old_output.shape[-1]))
            new_style_outputs.append(pca(old_output))
            pca.configure(feats)

        new_content_outputs = []
        for old_output, feats, in zip(feat_model.output['content'], feats_dict['content']):
            pca = PCA(min(FLAGS.pca, old_output.shape[-1]))
            new_content_outputs.append(pca(old_output))
            pca.configure(feats)

        new_feat_model = tf.keras.models.Model(feat_model.input,
                                               {'style': new_style_outputs, 'content': new_content_outputs})
        logging.info(f'features projected to {FLAGS.pca} maximum dimensions with PCA')

        sc_model.feat_model = new_feat_model
