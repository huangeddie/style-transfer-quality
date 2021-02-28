import tensorflow as tf
from absl import flags

import discriminators as disc

FLAGS = flags.FLAGS

flags.DEFINE_enum('feat_model', 'vgg19', ['vgg19', 'fast'],
                  'whether or not to cache the features when performing style transfer')
flags.DEFINE_enum('disc', 'm1', ['m1', 'm2', 'gram', 'm3'], 'type of discrimination to use')

class FlattenSpatial(tf.keras.layers.Layer):
    """
    Assumes channel last input
    """
    def call(self, inputs, **kwargs):
        tf.debugging.assert_rank(inputs, 4)
        input_shape = tf.shape(inputs)
        bsz, feat_dim = input_shape[0], input_shape[-1]
        return tf.reshape(inputs, [bsz, -1, feat_dim])


def load_feat_model(input_shape):
    flatten_spatial = FlattenSpatial()
    if FLAGS.feat_model == 'vgg19':
        style_input = tf.keras.Input(input_shape)
        content_input = tf.keras.Input(input_shape)

        preprocess_fn = tf.keras.applications.vgg19.preprocess_input
        vgg = tf.keras.applications.VGG19(include_top=False)
        vgg.trainable = False

        content_layers = ['block5_conv2']
        style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
        vgg_style_outputs = [flatten_spatial(vgg.get_layer(name).output) for name in style_layers]
        vgg_content_outputs = [flatten_spatial(vgg.get_layer(name).output) for name in content_layers]

        vgg_style = tf.keras.Model(vgg.input, vgg_style_outputs)
        vgg_content = tf.keras.Model(vgg.input, vgg_content_outputs)

        x = preprocess_fn(style_input)
        style_output = vgg_style(x)
        style_model = tf.keras.Model(style_input, style_output)

        x = preprocess_fn(content_input)
        content_output = vgg_content(x)
        content_model = tf.keras.Model(content_input, content_output)
    elif FLAGS.feat_model == 'fast':
        style_input = tf.keras.Input(input_shape)
        content_input = tf.keras.Input(input_shape)
        avg_pool = tf.keras.layers.AveragePooling2D(pool_size=4)

        style_model = tf.keras.Model(style_input, [flatten_spatial(avg_pool(style_input))])
        content_model = tf.keras.Model(content_input, [flatten_spatial(avg_pool(content_input))])
    else:
        raise ValueError(f'unknown feature model: {FLAGS.feat_model}')

    return tf.keras.Model([style_model.input, content_model.input],
                          {'style': style_model.outputs, 'content': content_model.outputs})


def make_discriminator():
    if FLAGS.disc == 'm1':
        return disc.FirstMomentLoss()
    elif FLAGS.disc == 'gram':
        return disc.GramianLoss()
    elif FLAGS.disc == 'm3':
        return disc.ThirdMomentLoss()
    else:
        raise ValueError(f'unknown discriminator: {FLAGS.disc}')


class SCModel(tf.keras.Model):
    def __init__(self, input_shape, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.feat_model = load_feat_model(input_shape)
        self.discriminator = make_discriminator()

    def build(self, input_shape):
        assert len(input_shape) == 2
        assert input_shape[0] == input_shape[1]
        self.gen_image = self.add_weight('gen_image', input_shape[0],
                                         initializer=tf.keras.initializers.RandomUniform(minval=0, maxval=255))

    def call(self, inputs, training=None, mask=None):
        return self.feat_model(inputs)

    def train_step(self, data):
        _, feats = data

        with tf.GradientTape() as tape:
            # Compute generated features
            gen_feats = self.feat_model((self.gen_image, self.gen_image))

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
