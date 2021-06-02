import paddle.nn as nn


def freeze_batchnorm_statictis(layer):
    def freeze_bn(layer):
        if isinstance(layer, (nn.layer.norm._BatchNormBase)):
            layer._use_global_stats = True

    layer.apply(freeze_bn)