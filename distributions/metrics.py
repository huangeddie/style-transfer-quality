from functools import partial

import tensorflow_addons as tfa

from distributions import compute_mean_loss, compute_var_loss, \
    compute_covar_loss, compute_raw_m2_loss, compute_skew_loss, compute_wass_dist


class MeanLoss(tfa.metrics.MeanMetricWrapper):
    def __init__(self, name="mean_loss", **kwargs):
        super().__init__(partial(compute_mean_loss, p=1), name=name, **kwargs)


class VarLoss(tfa.metrics.MeanMetricWrapper):
    def __init__(self, name="var_loss", **kwargs):
        super().__init__(partial(compute_var_loss, p=1), name=name, **kwargs)


class CovarLoss(tfa.metrics.MeanMetricWrapper):
    def __init__(self, name="covar_loss", **kwargs):
        super().__init__(partial(compute_covar_loss, p=1), name=name, **kwargs)


class GramLoss(tfa.metrics.MeanMetricWrapper):
    def __init__(self, name="gram_loss", **kwargs):
        super().__init__(partial(compute_raw_m2_loss, p=1), name=name, **kwargs)


class SkewLoss(tfa.metrics.MeanMetricWrapper):
    def __init__(self, name="skew_loss", **kwargs):
        super().__init__(partial(compute_skew_loss, p=1), name=name, **kwargs)


class WassDist(tfa.metrics.MeanMetricWrapper):
    def __init__(self, name="wass_dist", **kwargs):
        super().__init__(partial(compute_wass_dist, p=1), name=name, **kwargs)
