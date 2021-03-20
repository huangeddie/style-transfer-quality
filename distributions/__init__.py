import tensorflow as tf


def _flatten_spatial(x):
    shape = tf.shape(x)
    bsz, num_locs, channels = shape[0], shape[1] * shape[2], shape[3]
    x = tf.reshape(x, [bsz, num_locs, channels])
    return x


def get_p_fn(p):
    if p == 1:
        return tf.abs
    elif p == 2:
        return tf.square
    else:
        raise ValueError(f'unexpected p value: {p}')


def sample_k(x, k):
    if k is not None:
        x = tf.transpose(x, [1, 0, 2])
        n = tf.shape(x)[0]
        x = tf.gather(x, tf.random.shuffle(tf.range(n))[:tf.minimum(k, n)])
        x = tf.transpose(x, [1, 0, 2])
    return x

def process_spatial_feats(x, k):
    x = _flatten_spatial(x)
    x = sample_k(x, k)
    return x

def compute_wass_dist(y_true, y_pred, p):
    y, x = tf.sort(y_true, axis=1), tf.sort(y_pred, axis=1)
    p_fn = get_p_fn(p)
    wass_dist = tf.reduce_mean(p_fn(y - x), axis=1)
    return tf.reduce_mean(wass_dist, axis=-1)


def compute_mean_loss(y_true, y_pred, p):
    mu1 = tf.reduce_mean(y_true, axis=1)
    mu2 = tf.reduce_mean(y_pred, axis=1)
    p_fn = get_p_fn(p)
    mean_loss = tf.reduce_mean(p_fn(mu1 - mu2), axis=-1)
    return mean_loss


def compute_raw_m2_loss(y_true, y_pred, p):
    shape = tf.shape(y_true)
    num_locs = tf.cast(shape[1], y_true.dtype)
    raw_m2_1 = tf.einsum('bnc,bnd->bcd', y_true, y_true) / num_locs
    raw_m2_2 = tf.einsum('bnc,bnd->bcd', y_pred, y_pred) / num_locs
    p_fn = get_p_fn(p)
    raw_m2_loss = tf.reduce_mean(p_fn(raw_m2_1 - raw_m2_2), axis=1)
    return tf.reduce_mean(raw_m2_loss, axis=-1)


def compute_var_loss(y_true, y_pred, p):
    var1 = tf.math.reduce_variance(y_true, axis=1)
    var2 = tf.math.reduce_variance(y_pred, axis=1)

    p_fn = get_p_fn(p)
    var_loss = tf.reduce_mean(p_fn(var1 - var2), axis=-1)
    return var_loss


def compute_covar_loss(y_true, y_pred, p):
    mu1 = tf.reduce_mean(y_true, axis=1, keepdims=True)
    mu2 = tf.reduce_mean(y_pred, axis=1, keepdims=True)
    centered_y1 = y_true - mu1
    centered_y2 = y_pred - mu2
    covar_loss = compute_raw_m2_loss(centered_y1, centered_y2, p)
    return covar_loss


def compute_skew_loss(y_true, y_pred, p):
    mu1, var1 = tf.nn.moments(y_true, axes=1, keepdims=True)
    mu2, var2 = tf.nn.moments(y_pred, axes=1, keepdims=True)
    z1 = (y_true - mu1) * tf.math.rsqrt(var1 + 1e-3)
    z2 = (y_pred - mu2) * tf.math.rsqrt(var2 + 1e-3)

    skew1 = tf.reduce_mean(z1 ** 3, axis=1)
    skew2 = tf.reduce_mean(z2 ** 3, axis=1)
    p_fn = get_p_fn(p)
    skew_loss = tf.reduce_mean(p_fn(skew1 - skew2), axis=-1)
    return skew_loss
