import tensorflow as tf
from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_bool('cache_feats', True, 'whether or not to cache the features when performing style transfer')
flags.DEFINE_enum('feat_model', 'vgg19', ['vgg19'],
                  'whether or not to cache the features when performing style transfer')


def load_feat_model():
    if FLAGS.feat_model == 'vgg19':
        input = tf.keras.Input()
        x = tf.keras.applications.vgg19.preprocess_input(input)
        vgg = tf.keras.applications.VGG19(input_tensor=x, include_top=False)
        vgg.trainable = False

        content_layers = ['block5_conv2']
        style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
        style_outputs = [vgg.get_layer(name).output for name in style_layers]
        content_outputs = [vgg.get_layer(name).output for name in content_layers]
    else:
        raise ValueError(f'unknown feature model: {FLAGS.feat_model}')
    return tf.keras.Model(input, (style_outputs, content_outputs))


def make_discriminator():
    pass


class SCModel(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.feat_model = load_feat_model()
        self.discriminator = make_discriminator()

    def build(self, input_shape):
        self.gen_image = tf.random.uniform(input_shape)

    def cache_feats(self, style_image, content_image):
        self.style_feats = tf.constant(self.feat_model(style_image))
        self.content_feats = tf.constant(self.feat_model(content_image))

    def train_step(self, data):
        style_image, content_image = data

        # Get style and content features
        if hasattr(self, 'style_feats') and hasattr(self, 'content_feats'):
            style_feats, content_feats = self.style_feats, self.content_feats
        else:
            style_feats, content_feats = self.feat_model(style_image), self.feat_model(content_image)

        # Get generated features
        gen_feats = self.feat_model(self.gen_image)

        loss = 0
        # Discriminate between style and gen features

        # Discriminate between content and gen features

        # Gradient descent over the discrimination loss

        return {'loss': loss}

    def get_gen_image(self):
        return tf.constant(self.gen_image)
