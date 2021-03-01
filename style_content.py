import tensorflow as tf
from absl import flags

import dist_losses
import dist_metrics

FLAGS = flags.FLAGS

flags.DEFINE_enum('feat_model', 'vgg19', ['vgg19', 'nasnetlarge', 'fast'],
                  'whether or not to cache the features when performing style transfer')


class Preprocess(tf.keras.layers.Layer):
    def __init__(self, preprocess_fn, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.preprocess = preprocess_fn

    def call(self, inputs, **kwargs):
        return self.preprocess(inputs)


def load_feat_model(input_shape):
    if FLAGS.feat_model == 'vgg19':
        style_input = tf.keras.Input(input_shape)
        content_input = tf.keras.Input(input_shape)

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
        style_model = tf.keras.Model(style_input, style_output)

        x = preprocess_fn(content_input)
        content_output = vgg_content(x)
        content_model = tf.keras.Model(content_input, content_output)
    elif FLAGS.feat_model == 'nasnetlarge':
        style_input = tf.keras.Input(input_shape)
        content_input = tf.keras.Input(input_shape)

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
        style_model = tf.keras.Model(style_input, style_output)

        x = preprocess_fn(content_input)
        content_output = nasnet_content(x)
        content_model = tf.keras.Model(content_input, content_output)
    elif FLAGS.feat_model == 'fast':
        style_input = tf.keras.Input(input_shape)
        content_input = tf.keras.Input(input_shape)
        avg_pool = tf.keras.layers.AveragePooling2D(pool_size=4)

        style_model = tf.keras.Model(style_input, [avg_pool(style_input)])
        content_model = tf.keras.Model(content_input, [avg_pool(content_input)])
    else:
        raise ValueError(f'unknown feature model: {FLAGS.feat_model}')

    new_style_outputs = [tf.keras.layers.BatchNormalization(scale=False, center=False, momentum=0)(output) for
                         output in style_model.outputs]
    new_content_outputs = [tf.keras.layers.BatchNormalization(scale=False, center=False, momentum=0)(output) for
                           output in content_model.outputs]

    return tf.keras.Model([style_model.input, content_model.input],
                          {'style': new_style_outputs, 'content': new_content_outputs})


class SCModel(tf.keras.Model):
    def __init__(self, input_shape, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.feat_model = load_feat_model(input_shape)

    def build(self, input_shape):
        self.gen_image = self.add_weight('gen_image', input_shape[0],
                                         initializer=tf.keras.initializers.RandomUniform(minval=0, maxval=255))

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


def make_sc_model(strategy, image_shape, loss_key):
    with strategy.scope():
        sc_model = SCModel(image_shape)

        losses = {'style': [dist_losses.loss_dict[loss_key] for _ in sc_model.feat_model.output['style']]}
        metrics = {'style': [[dist_metrics.MeanLoss(), dist_metrics.VarLoss(), dist_metrics.SkewLoss()] for _ in
                             sc_model.feat_model.output['style']],
                   'content': [[] for _ in sc_model.feat_model.output['content']]}
        if FLAGS.content_image is not None:
            losses['content'] = [tf.keras.losses.MeanSquaredError() for _ in sc_model.feat_model.output['content']]

        sc_model.compile(tf.keras.optimizers.Adam(FLAGS.lr, FLAGS.beta1, FLAGS.beta2, FLAGS.epsilon), loss=losses,
                         metrics=metrics)
    return sc_model
