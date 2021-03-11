import tensorflow as tf


def compute_wass_dist(y_true, y_pred, p=1):
    shape = tf.shape(y_true)
    print([y_true.shape[i] for i in range(4)])
    print([y_pred.shape[i] for i in range(4)])
    bsz, num_locs, channels = shape[0], shape[1] * shape[2], shape[3]
    y_true = tf.reshape(y_true, [bsz, num_locs, channels])
    y_pred = tf.reshape(y_pred, [bsz, num_locs, channels])
    y, x = tf.sort(y_true, axis=1), tf.sort(y_pred, axis=1)
    if p == 1:
        wass_dist = tf.reduce_mean(tf.abs(y - x), axis=1)
    else:
        assert p == 2
        wass_dist = tf.reduce_mean(tf.square(y - x), axis=1)
    return tf.reduce_mean(wass_dist, axis=-1)


def compute_mean_loss(y_true, y_pred):
    mu1 = tf.reduce_mean(y_true, axis=[1, 2])
    mu2 = tf.reduce_mean(y_pred, axis=[1, 2])
    mean_loss = tf.reduce_mean((mu1 - mu2) ** 2, axis=1)
    return mean_loss


def compute_raw_m2_loss(x, y):
    shape = tf.shape(x)
    num_locs = tf.cast(shape[1] * shape[2], x.dtype)
    covar1 = tf.einsum('bhwc,bhwd->bcd', x, x) / num_locs
    covar2 = tf.einsum('bhwc,bhwd->bcd', y, y) / num_locs
    covar_loss = tf.reduce_mean((covar1 - covar2) ** 2, axis=1)
    return tf.reduce_mean(covar_loss, axis=-1)


def compute_var_loss(y_true, y_pred):
    var1 = tf.math.reduce_variance(y_true, axis=[1, 2])
    var2 = tf.math.reduce_variance(y_pred, axis=[1, 2])

    var_loss = tf.reduce_mean((var1 - var2) ** 2, axis=1)
    return var_loss


def compute_covar_loss(y_true, y_pred):
    mu1 = tf.reduce_mean(y_true, axis=[1, 2], keepdims=True)
    mu2 = tf.reduce_mean(y_pred, axis=[1, 2], keepdims=True)
    centered_y1 = y_true - mu1
    centered_y2 = y_pred - mu2
    covar_loss = compute_raw_m2_loss(centered_y1, centered_y2)
    return covar_loss


def compute_skew_loss(y_pred, y_true):
    mu1, var1 = tf.nn.moments(y_true, axes=[1, 2], keepdims=True)
    mu2, var2 = tf.nn.moments(y_pred, axes=[1, 2], keepdims=True)
    z1 = (y_true - mu1) * tf.math.rsqrt(var1 + 1e-3)
    z2 = (y_pred - mu2) * tf.math.rsqrt(var2 + 1e-3)
    skew1 = tf.reduce_mean(z1 ** 3, axis=[1, 2])
    skew2 = tf.reduce_mean(z2 ** 3, axis=[1, 2])
    skew_loss = tf.reduce_mean((skew1 - skew2) ** 2, axis=1)
    return skew_loss
