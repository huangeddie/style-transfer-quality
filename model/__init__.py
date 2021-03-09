import tensorflow as tf
from absl import flags
from absl import logging

from model.layers import Preprocess, Standardize, PCA, FastICA

FLAGS = flags.FLAGS

flags.DEFINE_enum('start_image', 'rand', ['rand', 'black'], 'image size')

flags.DEFINE_enum('feat_model', 'vgg19', ['vgg19', 'nasnetlarge', 'fast'], 'feature model architecture')
flags.DEFINE_integer('layers', 5, 'number of layers to use from the feature model')
flags.DEFINE_enum('disc_model', None, ['mlp', 'fast'], 'discriminator model architecture')
flags.DEFINE_float('disc_l2', 0, 'l2 reg for discriminator')

flags.DEFINE_bool('shift', False, 'standardize outputs based on the style & content features')
flags.DEFINE_bool('scale', False, 'standardize outputs based on the style & content features')

flags.DEFINE_integer('pca', None, 'maximum dimension of features enforced with PCA')
flags.DEFINE_integer('ica', None, 'maximum dimension of features enforced with FastICa')
flags.DEFINE_bool('whiten', False, 'whiten the components of PCA/ICA')


def make_feat_model(input_shape):
    style_input = tf.keras.Input(input_shape, name='style')
    content_input = tf.keras.Input(input_shape, name='content')
    if FLAGS.feat_model == 'vgg19':
        preprocess_fn = Preprocess(tf.keras.applications.vgg19.preprocess_input)
        vgg19 = tf.keras.applications.VGG19(include_top=False)
        vgg19.trainable = False

        content_layers = ['block5_conv2']
        style_layers = [f'block{i}_conv1' for i in range(1, FLAGS.layers + 1)]
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
        for layer in [avg_pool1, avg_pool2][:FLAGS.layers]:
            x = layer(x)
            style_output.append(x)

        x = content_input
        content_output = []
        for layer in [avg_pool1, avg_pool2][:FLAGS.layers]:
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


def make_discriminator(feat_model):
    if FLAGS.disc_model is None:
        return None

    inputs, outputs = [], []
    for style_output in feat_model.output['style']:
        l2_reg = tf.keras.regularizers.L2(FLAGS.disc_l2)
        input_shape = style_output.shape[1:]
        feat_dim = input_shape[-1]
        hdim = min(max(2 * feat_dim, 64), 512)
        if FLAGS.disc_model == 'fast':
            layer_disc = tf.keras.Sequential([
                tf.keras.layers.Dense(1, kernel_regularizer=l2_reg, bias_regularizer=l2_reg)
            ])
        elif FLAGS.disc_model == 'mlp':
            layer_disc = tf.keras.Sequential([
                tf.keras.layers.Dense(hdim, kernel_regularizer=l2_reg, bias_regularizer=l2_reg),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
                tf.keras.layers.Dense(hdim, kernel_regularizer=l2_reg, bias_regularizer=l2_reg),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
                tf.keras.layers.Dense(1, kernel_regularizer=l2_reg, bias_regularizer=l2_reg),
            ])
        else:
            raise ValueError(f'unknown discriminator model: {FLAGS.disc_model}')
        input = tf.keras.Input(input_shape)
        output = layer_disc(input)
        inputs.append(input)
        outputs.append(output)
    return tf.keras.Model(inputs, outputs)


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
        logging.info(f'initialzed gen image with {initializer.__class__.__name__}')
        self.gen_image = self.add_weight('gen_image', input_shape[0], initializer=initializer)

    def configure(self, style_image, content_image):
        feat_model = self.feat_model

        # Configure the standardize layers if any
        # Standardize layers before building the generated image
        # or else the standardize layers will be configured on the gen image
        logging.info(f'configuring standardize layers (shift={FLAGS.shift}, scale={FLAGS.scale})')
        feats_dict = feat_model((style_image, content_image))

        # Build the gen image
        self((style_image, content_image))

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

            self.feat_model = new_feat_model

        # Add discriminator if requested
        if FLAGS.disc_model is not None:
            self.discriminator = make_discriminator(self.feat_model)
            logging.info('added discriminator')

    def reinit_gen_image(self):
        self.gen_image.assign(tf.random.uniform(self.gen_image.shape, maxval=255, dtype=self.gen_image.dtype))

    def call(self, inputs, training=None, mask=None):
        return self.feat_model((self.gen_image, self.gen_image), training=training)

    def train_step(self, data):
        images, feats = data

        # Train the discriminator
        if hasattr(self, 'discriminator'):
            d_cost, gp, d_grads, d_weights = self.disc_step(images, feats)
        else:
            d_cost, gp, d_grads, d_weights = None, None, [], []

        # Train the generated image
        g_grads, g_weights = self.gen_step(images, feats)

        self.optimizer.apply_gradients(zip(d_grads + g_grads, d_weights + g_weights))

        # Clip to RGB range
        self.gen_image.assign(tf.clip_by_value(self.gen_image, 0, 255))

        # Return a dict mapping metric names to current value + the discriminator loss
        return {**{m.name: m.result() for m in self.metrics}, 'd_cost': d_cost, 'gp': gp}

    def gen_step(self, images, feats):
        with tf.GradientTape() as tape:
            # Compute generated features
            gen_feats = self(images, training=False)

            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = self.compiled_loss(feats, gen_feats, regularization_losses=self.losses)

            # Add discriminator loss if any
            if hasattr(self, 'discriminator'):
                d_logits = self.discriminator(gen_feats['style'])
                if isinstance(d_logits, list):
                    gen_loss = [tf.reduce_mean(logits) for logits in d_logits]
                    gen_loss = tf.reduce_mean(gen_loss)
                else:
                    gen_loss = tf.reduce_mean(d_logits)
                loss += gen_loss
        # Optimize generated image
        grad = tape.gradient(loss, [self.gen_image])

        # Update metrics
        self.compiled_metrics.update_state(feats, gen_feats)
        return grad, [self.gen_image]

    def gradient_penalty(self, all_real_feats, all_fake_feats):
        """ Calculates the gradient penalty.

        This loss is calculated on an interpolated image
        and added to the discriminator loss.
        """
        # Get the interpolated image
        all_interpolated = []
        for real_feats, fake_feats in zip(all_real_feats, all_fake_feats):
            shape = tf.shape(real_feats)
            b, h, w, c = [shape[i] for i in range(4)]
            alpha = tf.random.uniform([b, h, w, 1])
            interpolated = alpha * real_feats + (1 - alpha) * fake_feats
            all_interpolated.append(interpolated)

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(all_interpolated)
            # 1. Get the discriminator output for this interpolated image.
            d_out = self.discriminator(all_interpolated)

        # 2. Calculate the gradients w.r.t to this interpolated image.
        grads = gp_tape.gradient(d_out, all_interpolated)
        # 3. Calculate the norm of the gradients.
        norms = [tf.norm(g, axis=-1) for g in grads]
        gps = [tf.reduce_mean((n - 1) ** 2) for n in norms]
        return gps

    def disc_step(self, images, feats):
        gen_feats = self(images, training=False)
        with tf.GradientTape() as tape:
            real_logits = self.discriminator(feats['style'])
            gen_logits = self.discriminator(gen_feats['style'])
            gps = self.gradient_penalty(feats['style'], gen_feats['style'])
            gps = tf.reduce_sum(gps)
            if isinstance(real_logits, list):
                d_costs = [tf.reduce_mean(rl - gl) for rl, gl in zip(real_logits, gen_logits)]
                d_costs = tf.reduce_sum(d_costs)
            else:
                d_costs = tf.reduce_mean(real_logits - gen_logits)
            d_loss = d_costs + 10 * gps
        d_grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        return d_costs, gps, d_grads, self.discriminator.trainable_weights

    def get_gen_image(self):
        return tf.constant(tf.cast(self.gen_image, tf.uint8))
