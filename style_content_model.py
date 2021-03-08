import tensorflow as tf
from absl import flags
from absl import logging
from sklearn import decomposition

FLAGS = flags.FLAGS

flags.DEFINE_enum('start_image', 'rand', ['rand', 'black'], 'image size')

flags.DEFINE_enum('feat_model', 'vgg19', ['vgg19', 'nasnetlarge', 'fast'],
                  'whether or not to cache the features when performing style transfer')

flags.DEFINE_bool('shift', False, 'standardize outputs based on the style & content features')
flags.DEFINE_bool('scale', False, 'standardize outputs based on the style & content features')

flags.DEFINE_integer('pca', None, 'maximum dimension of features enforced with PCA')
flags.DEFINE_integer('ica', None, 'maximum dimension of features enforced with FastICa')
flags.DEFINE_bool('whiten', False, 'whiten the components of PCA/ICA')

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


class Standardize(tf.keras.layers.Layer):
    def __init__(self, shift=True, scale=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.shift, self.scale = shift, scale

    def build(self, input_shape):
        feat_dim = input_shape[-1]
        self.mean = self.add_weight('mean', [1, 1, 1, feat_dim], trainable=False, initializer='zeros')
        self.variance = self.add_weight('variance', [1, 1, 1, feat_dim], trainable=False, initializer='ones')
        self.configured = self.add_weight('configured', [], trainable=False, dtype=tf.bool, initializer='zeros')

    def configure(self, feats):
        # Precision errors with float32
        feats = tf.cast(feats, tf.float64)

        if self.shift:
            self.mean.assign(tf.cast(tf.reduce_mean(feats, axis=[0, 1, 2], keepdims=True), tf.float32))

        if self.scale:
            self.variance.assign(tf.cast(tf.math.reduce_variance(feats, axis=[0, 1, 2], keepdims=True), tf.float32))

    def call(self, inputs, **kwargs):
        if self.configured == tf.zeros_like(self.configured):
            self.configure(inputs)
            self.configured.assign(tf.ones_like(self.configured))
        return (inputs - self.mean) * tf.math.rsqrt(self.variance + 1e-5)


class PCA(tf.keras.layers.Layer):
    def __init__(self, out_dim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.out_dim = out_dim

    def build(self, input_shape):
        feat_dim = input_shape[-1]
        self.mean = self.add_weight('mean', [1, 1, 1, feat_dim], trainable=False)
        self.projection = self.add_weight('projection', [feat_dim, self.out_dim], trainable=False)

    def configure(self, feats):
        # Precision errors with float32
        feats = tf.cast(feats, tf.float64)
        self.mean.assign(tf.cast(tf.reduce_mean(feats, axis=[0, 1, 2], keepdims=True), tf.float32))

        pca = decomposition.PCA(n_components=self.out_dim, whiten=FLAGS.whiten)
        feats_shape = tf.shape(feats)
        n_samples, feat_dim = tf.reduce_prod(feats_shape[:-1]), feats_shape[-1]

        pca.fit(tf.reshape(feats, [n_samples, feat_dim]))
        tf.debugging.assert_equal(tf.squeeze(self.mean), tf.constant(pca.mean_, dtype=tf.float32))
        self.projection.assign(tf.constant(pca.components_.T, dtype=self.projection.dtype))

    def call(self, inputs, **kwargs):
        x = inputs - self.mean
        components = tf.einsum('bhwc,cd->bhwd', x, self.projection)
        return tf.concat([inputs, components], axis=-1)


class FastICA(tf.keras.layers.Layer):
    def __init__(self, out_dim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.out_dim = out_dim

    def build(self, input_shape):
        feat_dim = input_shape[-1]
        self.mean = self.add_weight('mean', [1, 1, 1, feat_dim], trainable=False)
        self.projection = self.add_weight('projection', [feat_dim, self.out_dim], trainable=False)

    def configure(self, feats):
        # Precision errors with float32
        feats = tf.cast(feats, tf.float64)
        self.mean.assign(tf.cast(tf.reduce_mean(feats, axis=[0, 1, 2], keepdims=True), tf.float32))

        ica = decomposition.FastICA(n_components=self.out_dim)
        feats_shape = tf.shape(feats)
        n_samples, feat_dim = tf.reduce_prod(feats_shape[:-1]), feats_shape[-1]

        ica.fit(tf.reshape(feats, [n_samples, feat_dim]))
        tf.debugging.assert_equal(tf.squeeze(self.mean), tf.constant(ica.mean_, dtype=tf.float32))
        self.projection.assign(tf.constant(ica.components_.T, dtype=self.projection.dtype))

    def call(self, inputs, **kwargs):
        x = inputs - self.mean
        components = tf.einsum('bhwc,cd->bhwd', x, self.projection)
        return tf.concat([inputs, components], axis=-1)


class SCModel(tf.keras.Model):
    def __init__(self, feat_model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.feat_model = feat_model

    def build(self, input_shape):
        if FLAGS.start_image == 'rand':
            initializer = tf.keras.initializers.RandomUniform(minval=0, maxval=255)
        else:
            assert FLAGS.start_image == 'black'
            initializer = tf.keras.initializers.Zeros()
        logging.info(f'image initializer: {initializer.__class__.__name__}')
        self.gen_image = self.add_weight('gen_image', input_shape[0], initializer=initializer)

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
        avg_pool1 = tf.keras.layers.AveragePooling2D(pool_size=2)
        avg_pool2 = tf.keras.layers.AveragePooling2D(pool_size=2)

        x = style_input
        style_output = []
        for layer in [avg_pool1, avg_pool2]:
            x = layer(x)
            style_output.append(x)

        x = content_input
        content_output = []
        for layer in [avg_pool1, avg_pool2]:
            x = layer(x)
            content_output.append(x)

    else:
        raise ValueError(f'unknown feature model: {FLAGS.feat_model}')

    style_model = tf.keras.Model(style_input, style_output)
    content_model = tf.keras.Model(content_input, content_output)
    if FLAGS.shift or FLAGS.scale:
        new_style_outputs = [Standardize(FLAGS.shift, FLAGS.scale)(output) for output in style_model.outputs]
        new_content_outputs = [Standardize(FLAGS.shift, FLAGS.scale)(output) for output in content_model.outputs]

        sc_model = tf.keras.Model([style_model.input, content_model.input],
                                  {'style': new_style_outputs, 'content': new_content_outputs})
        logging.info('standardizing features')
    else:
        sc_model = tf.keras.Model([style_model.input, content_model.input],
                                  {'style': style_model.outputs, 'content': content_model.outputs})
    return sc_model


def configure_feat_model(sc_model, style_image, content_image):
    feat_model = sc_model.feat_model

    # Configure the standardize layers if any
    # Standardize layers before building the generated image
    # or else the standardize layers will be configured on the gen image
    logging.info('configuring standardize layers')
    feats_dict = feat_model((style_image, content_image))

    # Build the gen image
    logging.info('building gen image')
    sc_model((style_image, content_image))

    # Add and configure the PCA layers if requested
    if (FLAGS.pca is not None and FLAGS.pca > 0) or (FLAGS.ica is not None and FLAGS.ica > 0):
        ProjClass = PCA if FLAGS.pca is not None else FastICA
        proj_dim = FLAGS.pca or FLAGS.ica
        all_new_outputs = []

        for key in ['style', 'content']:
            new_outputs = []
            for old_output, feats, in zip(feat_model.output[key], feats_dict[key]):
                n_samples = old_output.shape[1] * old_output.shape[2]
                n_features = old_output.shape[-1]
                proj = ProjClass(min(proj_dim, n_features, n_samples))
                new_outputs.append(proj(old_output))
                proj.configure(feats)
            all_new_outputs.append(new_outputs)

        new_feat_model = tf.keras.models.Model(feat_model.input,
                                               {'style': all_new_outputs[0], 'content': all_new_outputs[1]})
        logging.info(f'features projected to {proj_dim} maximum dimensions with {ProjClass.__name__}')

        sc_model.feat_model = new_feat_model
