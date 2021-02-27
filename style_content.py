import tensorflow as tf
from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_bool("cache_feats", True, "whether or not to cache the features when performing style transfer")


class SCModel(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cache_feats = FLAGS.cache_feats

    def build(self, input_shape):
        self.gen_image = tf.random.uniform(input_shape)

    def train_step(self, data):
        print(f'data: {data}')
        pass
