import tensorflow_addons as tfa

from distributions import compute_mean_loss, compute_var_loss, \
    compute_covar_loss, compute_raw_m2_loss, compute_skew_loss, compute_wass_dist


class MeanLoss(tfa.metrics.MeanMetricWrapper):
    def __init__(self, name="mean_loss", **kwargs):
        super().__init__(compute_mean_loss, name=name, **kwargs)


class VarLoss(tfa.metrics.MeanMetricWrapper):
    def __init__(self, name="var_loss", **kwargs):
        super().__init__(compute_var_loss, name=name, **kwargs)


class CovarLoss(tfa.metrics.MeanMetricWrapper):
    def __init__(self, name="covar_loss", **kwargs):
        super().__init__(compute_covar_loss, name=name, **kwargs)


class GramLoss(tfa.metrics.MeanMetricWrapper):
    def __init__(self, name="gram_loss", **kwargs):
        super().__init__(compute_raw_m2_loss, name=name, **kwargs)


class SkewLoss(tfa.metrics.MeanMetricWrapper):
    def __init__(self, name="skew_loss", **kwargs):
        super().__init__(compute_skew_loss, name=name, **kwargs)


class WassDist(tfa.metrics.MeanMetricWrapper):
    def __init__(self, name="wass_dist", **kwargs):
        super().__init__(compute_wass_dist, name=name, **kwargs)
