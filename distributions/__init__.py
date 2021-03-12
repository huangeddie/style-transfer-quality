import tensorflow as tf


def sample_k(x, k):
    if k is not None:
        x = tf.random.shuffle(x)
        x = x[:k]
    return x


def flatten_spatial(x):
    shape = tf.shape(x)
    bsz, num_locs, channels = shape[0], shape[1] * shape[2], shape[3]
    x = tf.reshape(x, [bsz, num_locs, channels])
    return x


def preprocess_feats(x, k):
    x = flatten_spatial(x)
    x = tf.transpose(x, [1, 0, 2])
    x = sample_k(x, k)
    return x


def compute_wass_dist(y_true, y_pred, k=None, p=1):
    y_true, y_pred = preprocess_feats(y_true, k), preprocess_feats(y_pred, k)

    y, x = tf.sort(y_true, axis=0), tf.sort(y_pred, axis=0)
    if p == 1:
        wass_dist = tf.reduce_mean(tf.abs(y - x), axis=0)
    else:
        assert p == 2
        wass_dist = tf.reduce_mean(tf.square(y - x), axis=0)
    return tf.reduce_mean(wass_dist, axis=-1)


def compute_mean_loss(y_true, y_pred, k=None):
    y_true, y_pred = preprocess_feats(y_true, k), preprocess_feats(y_pred, k)

    mu1 = tf.reduce_mean(y_true, axis=0)
    mu2 = tf.reduce_mean(y_pred, axis=0)
    mean_loss = tf.reduce_mean((mu1 - mu2) ** 2, axis=-1)
    return mean_loss


def compute_raw_m2_loss(y_true, y_pred, k=None):
    y_true, y_pred = preprocess_feats(y_true, k), preprocess_feats(y_pred, k)
    shape = tf.shape(y_true)
    num_locs = tf.cast(shape[1] * shape[2], y_true.dtype)
    covar1 = tf.einsum('nbc,nbd->bcd', y_true, y_true) / num_locs
    covar2 = tf.einsum('nbc,nbd->bcd', y_pred, y_pred) / num_locs
    covar_loss = tf.reduce_mean((covar1 - covar2) ** 2, axis=1)
    return tf.reduce_mean(covar_loss, axis=-1)


def compute_var_loss(y_true, y_pred, k=None):
    y_true, y_pred = preprocess_feats(y_true, k), preprocess_feats(y_pred, k)

    var1 = tf.math.reduce_variance(y_true, axis=0)
    var2 = tf.math.reduce_variance(y_pred, axis=0)

    var_loss = tf.reduce_mean((var1 - var2) ** 2, axis=-1)
    return var_loss


def compute_covar_loss(y_true, y_pred, k=None):
    mu1 = tf.reduce_mean(y_true, axis=[1, 2], keepdims=True)
    mu2 = tf.reduce_mean(y_pred, axis=[1, 2], keepdims=True)
    centered_y1 = y_true - mu1
    centered_y2 = y_pred - mu2
    covar_loss = compute_raw_m2_loss(centered_y1, centered_y2, k)
    return covar_loss


def compute_skew_loss(y_true, y_pred, k=None):
    mu1, var1 = tf.nn.moments(y_true, axes=[1, 2], keepdims=True)
    mu2, var2 = tf.nn.moments(y_pred, axes=[1, 2], keepdims=True)
    z1 = (y_true - mu1) * tf.math.rsqrt(var1 + 1e-3)
    z2 = (y_pred - mu2) * tf.math.rsqrt(var2 + 1e-3)

    z1, z2 = preprocess_feats(z1, k), preprocess_feats(z2, k)

    skew1 = tf.reduce_mean(z1 ** 3, axis=0)
    skew2 = tf.reduce_mean(z2 ** 3, axis=0)
    skew_loss = tf.reduce_mean((skew1 - skew2) ** 2, axis=-1)
    return skew_loss
