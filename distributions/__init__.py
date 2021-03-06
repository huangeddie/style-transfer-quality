import tensorflow as tf


def compute_wass_dist(y_pred, y_true, p=1):
    shape = tf.shape(y_true)
    bsz, num_locs, channels = shape[0], shape[1] * shape[2], shape[-1]
    y_true = tf.reshape(y_true, [bsz, num_locs, channels])
    y_pred = tf.reshape(y_pred, [bsz, num_locs, channels])
    y, x = tf.sort(y_true, axis=1), tf.sort(y_pred, axis=1)
    if p == 1:
        wass_dist = tf.reduce_mean(tf.abs(y - x), axis=1)
    else:
        assert p == 2
        wass_dist = tf.reduce_mean(tf.square(y - x), axis=1)
    return wass_dist


def reshape_to_feats(y_true, y_pred):
    y_shape = tf.shape(y_true)
    b, h, w, c = [y_shape[i] for i in range(4)]
    feats1 = tf.reshape(y_true, [b, h * w, c])
    feats2 = tf.reshape(y_pred, [b, h * w, c])
    return feats1, feats2


def compute_raw_m2_loss(x, y):
    shape = tf.shape(x)
    num_locs = tf.cast(shape[1] * shape[2], x.dtype)
    covar1 = tf.einsum('bhwc,bhwd->bcd', x, x) / num_locs
    covar2 = tf.einsum('bhwc,bhwd->bcd', y, y) / num_locs
    covar_loss = tf.reduce_mean((covar1 - covar2) ** 2, axis=[1, 2])
    return covar_loss