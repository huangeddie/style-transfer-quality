import tensorflow as tf


def compute_wass_dist(y_pred, y_true):
    shape = tf.shape(y_true)
    bsz, num_locs, channels = shape[0], shape[1] * shape[2], shape[-1]
    y_true = tf.reshape(y_true, [bsz, num_locs, channels])
    y_pred = tf.reshape(y_pred, [bsz, num_locs, channels])
    y, x = tf.sort(y_true, axis=1), tf.sort(y_pred, axis=1)
    wass_dist = tf.reduce_mean(tf.abs(y - x), axis=1)
    return wass_dist