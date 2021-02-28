import tensorflow as tf
from absl import flags

import discriminators as disc

FLAGS = flags.FLAGS

flags.DEFINE_bool('cache_feats', True, 'whether or not to cache the features when performing style transfer')
flags.DEFINE_enum('feat_model', 'vgg19', ['vgg19', 'fast'],
                  'whether or not to cache the features when performing style transfer')
flags.DEFINE_enum('disc', 'bn', ['bn', 'gram', 'm3'], 'type of discrimination to use')


def load_feat_model(input_shape):
    if FLAGS.feat_model == 'vgg19':
        input = tf.keras.Input(input_shape)
        x = tf.keras.applications.vgg19.preprocess_input(input)
        vgg = tf.keras.applications.VGG19(input_tensor=x, include_top=False)
        vgg.trainable = False

        content_layers = ['block5_conv2']
        style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
        style_outputs = [vgg.get_layer(name).output for name in style_layers]
        content_outputs = [vgg.get_layer(name).output for name in content_layers]
    elif FLAGS.feat_model == 'fast':
        input = tf.keras.Input(input_shape)
        x = tf.keras.layers.AveragePooling2D(pool_size=32)(input)
        style_outputs, content_outputs = [x], [x]
    else:
        raise ValueError(f'unknown feature model: {FLAGS.feat_model}')
    return tf.keras.Model(input, (style_outputs, content_outputs))


def make_discriminator():
    if FLAGS.disc == 'bn':
        return disc.BatchNormDiscriminator()
    elif FLAGS.disc == 'gram':
        return disc.GramianDiscriminator()
    elif FLAGS.disc == 'm3':
        return disc.ThirdMomentDiscriminator()
    else:
        raise ValueError(f'unknown discriminator: {FLAGS.disc}')


class SCModel(tf.keras.Model):
    def __init__(self, input_shape, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.feat_model = load_feat_model(input_shape)
        self.discriminator = make_discriminator()

    def build(self, input_shape):
        assert isinstance(input_shape, tuple)
        assert len(input_shape) == 2
        image_shape = input_shape[0]
        assert image_shape == input_shape[1]
        assert len(image_shape) == 4
        self.gen_image = self.add_weight('gen_image', image_shape, initializer=tf.keras.initializers.random_uniform)

    def call(self, inputs, training=None, mask=None):
        style_image, content_image = inputs
        style_feats, _ = self.feat_model(style_image)
        _, content_feats = self.feat_model(style_image)
        return style_feats, content_feats

    def cache_feats(self, style_image, content_image):
        style_feats, content_feats = self((style_image, content_image))
        self.style_feats = [tf.constant(feats) for feats in style_feats]
        self.content_feats = [tf.constant(feats) for feats in content_feats]

    def train_step(self, data):

        # Get style and content features
        if hasattr(self, 'style_feats') and hasattr(self, 'content_feats'):
            style_feats, content_feats = self.style_feats, self.content_feats
        else:
            style_feats, content_feats = self(data)

        with tf.GradientTape() as tape:
            # Compute generated features
            gen_style_feats, gen_content_feats = self.feat_model(self.gen_image)

            # Discriminate between style and gen features
            style_loss = self.discriminator((style_feats, gen_style_feats))

            # Discriminate between content and gen features
            if FLAGS.content_image is not None:
                content_loss = tf.keras.losses.mean_squared_error(gen_content_feats, )
            else:
                content_loss = 0

            # Gradient descent over the discrimination loss
            style_loss = tf.nn.compute_average_loss(style_loss, global_batch_size=FLAGS.bsz)
            content_loss = tf.nn.compute_average_loss(content_loss, global_batch_size=FLAGS.bsz)
            loss = style_loss + content_loss

        # Optimize generated image
        grad = tape.gradient(loss, self.gen_image.trainable_weights)
        self.optimizer.apply_gradients(zip(grad, self.gen_image.trainable_weights))

        # Train discriminator
        d_metrics = self.discriminator.train_step((style_feats, gen_style_feats))

        # Update metrics
        self.compiled_metrics.update_state((style_feats, content_feats), (gen_style_feats, gen_content_feats))

        # Return a dict mapping metric names to current value
        return {'loss': loss, 'style_loss': style_loss, 'content_loss': content_loss,
                **d_metrics, **{m.name: m.result() for m in self.metrics}}

    def get_gen_image(self):
        return tf.constant(self.gen_image)
