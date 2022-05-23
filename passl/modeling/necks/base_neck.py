# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle
import paddle.nn as nn
import paddle.fluid.layers as layers

from .builder import NECKS
from paddle.vision.models.resnet import BasicBlock, BottleneckBlock
from ...modules.init import init_backbone_weight, normal_init, kaiming_init, constant_, reset_parameters, xavier_init, init_backbone_weight_simclr


def _init_parameters(module, init_linear='normal', std=0.01, bias=0.):
    assert init_linear in ['normal', 'kaiming'], \
        "Undefined init_linear: {}".format(init_linear)
    for m in module.sublayers():
        if isinstance(m, nn.Linear):
            if init_linear == 'normal':
                normal_init(m, std=std, bias=bias)
            else:
                kaiming_init(m, mode='fan_in', nonlinearity='relu')
        elif isinstance(
                m,
            (nn.BatchNorm1D, nn.BatchNorm2D, nn.GroupNorm, nn.SyncBatchNorm)):
            if m.weight is not None:
                constant_(m.weight, 1)
            if m.bias is not None:
                constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2D):
            kaiming_init(m, mode='fan_in', nonlinearity='relu')


@NECKS.register()
class LinearNeck(nn.Layer):
    """Linear neck: fc only.
    """

    def __init__(self, in_channels, out_channels, with_avg_pool=True):
        super(LinearNeck, self).__init__()
        self.with_avg_pool = with_avg_pool
        if with_avg_pool:
            self.avgpool = nn.AdaptiveAvgPool2D((1, 1))
        self.fc = nn.Linear(in_channels, out_channels)
        # init_backbone_weight(self.fc)
        self.init_parameters()

    def init_parameters(self, init_linear='kaiming'):
        _init_parameters(self, init_linear)

    def forward(self, x):

        if self.with_avg_pool:
            x = self.avgpool(x)
        return self.fc(x.reshape([x.shape[0], -1]))


@NECKS.register()
class NonLinearNeckV1(nn.Layer):
    """The non-linear neck in MoCo v2: fc-relu-fc.
    """

    def __init__(self,
                 in_channels,
                 hid_channels,
                 out_channels,
                 with_avg_pool=True):
        super(NonLinearNeckV1, self).__init__()
        self.with_avg_pool = with_avg_pool
        if with_avg_pool:
            self.avgpool = nn.AdaptiveAvgPool2D((1, 1))

        self.mlp = nn.Sequential(nn.Linear(in_channels,
                                           hid_channels), nn.ReLU(),
                                 nn.Linear(hid_channels, out_channels))

        # init_backbone_weight(self.mlp)
        self.init_parameters()

    def init_parameters(self, init_linear='kaiming'):
        _init_parameters(self, init_linear)

    def forward(self, x):

        if self.with_avg_pool:
            x = self.avgpool(x)
        return self.mlp(x.reshape([x.shape[0], -1]))


@NECKS.register()
class NonLinearNeckV2(nn.Layer):
    """The non-linear neck in MoCo v2: fc-relu-fc.
    """

    def __init__(self,
                 in_channels,
                 hid_channels,
                 out_channels,
                 with_avg_pool=True):
        super(NonLinearNeckV2, self).__init__()
        self.with_avg_pool = with_avg_pool
        if with_avg_pool:
            self.avgpool = nn.AdaptiveAvgPool2D((1, 1))

        self.mlp = nn.Sequential(nn.Linear(in_channels, hid_channels),
                                 nn.BatchNorm1D(hid_channels), nn.ReLU(),
                                 nn.Linear(hid_channels, out_channels))

        # init_backbone_weight(self.mlp)
        # self.init_parameters()

    def init_parameters(self, init_linear='kaiming'):
        # _init_parameters(self, init_linear)
        for m in self.sublayers():
            if isinstance(m, nn.Linear):
                xavier_init(m, distribution='uniform')
            elif isinstance(m, (nn.BatchNorm1D, nn.BatchNorm2D, nn.GroupNorm,
                                nn.SyncBatchNorm)):
                if m.weight is not None:
                    constant_(m.weight, 1)
                if m.bias is not None:
                    constant_(m.bias, 0)

    def forward(self, x):
        if self.with_avg_pool:
            x = self.avgpool(x)
        return self.mlp(x.reshape([x.shape[0], -1]))


@NECKS.register()
class NonLinearNeckV3(nn.Layer):
    """MLP"""

    def __init__(self, in_channels, hid_channels, out_channels):
        super(NonLinearNeckV3, self).__init__()

        self.l1 = nn.Linear(in_channels, hid_channels)
        self.bn1 = nn.BatchNorm1D(hid_channels)
        self.relu1 = nn.ReLU()
        self.l2 = nn.Linear(hid_channels, out_channels)

    def init_parameters(self, init_linear='kaiming'):
        # _init_parameters(self, init_linear)
        for m in self.sublayers():
            if isinstance(m, nn.Linear):
                xavier_init(m, distribution='uniform')
            elif isinstance(m, (nn.BatchNorm1D, nn.BatchNorm2D, nn.GroupNorm,
                                nn.SyncBatchNorm)):
                if m.weight is not None:
                    constant_(m.weight, 1)
                if m.bias is not None:
                    constant_(m.bias, 0)

    def forward(self, x):
        """forward"""
        x = self.l1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.l2(x)
        return x


@NECKS.register()
class ConvNonLinearNeck(nn.Layer):
    """
    The Convolutioanl Neck proposed by F.
    """

    def __init__(self,
                 in_channels,
                 hid_channels,
                 out_channels,
                 with_avg_pool=True):
        super(ConvNonLinearNeck, self).__init__()
        self.with_avg_pool = with_avg_pool
        assert with_avg_pool, 'The with_avg_pool must be set to True in ConvNonLinearNeck!'
        if with_avg_pool:
            self.avgpool = nn.AdaptiveAvgPool2D((1, 1))

        self.conv = BottleneckBlock(in_channels, in_channels // 4)

        self.mlp = nn.Sequential(nn.Linear(in_channels,
                                           hid_channels), nn.ReLU(),
                                 nn.Linear(hid_channels, out_channels))

        init_backbone_weight(self.mlp)

    def init_parameters(self, init_linear='normal'):
        _init_parameters(self, init_linear)

    def forward(self, x):
        x = self.conv(x)
        if self.with_avg_pool:
            x = self.avgpool(x)
        return self.mlp(x.reshape([x.shape[0], -1]))


@NECKS.register()
class NonLinearNeckfc3(nn.Layer):
    """The non-linear neck in MoCo v2: fc-relu-fc-relu-fc.
    """

    def __init__(self,
                 in_channels,
                 hid_channels,
                 out_channels,
                 with_avg_pool=True):
        super(NonLinearNeckfc3, self).__init__()
        self.with_avg_pool = with_avg_pool
        if with_avg_pool:
            self.avgpool = nn.AdaptiveAvgPool2D((1, 1))
        self.mlp = nn.Sequential(nn.Linear(in_channels, hid_channels),
                                 nn.BatchNorm1D(hid_channels), nn.ReLU(),
                                 nn.Linear(hid_channels, hid_channels),
                                 nn.BatchNorm1D(hid_channels), nn.ReLU(),
                                 nn.Linear(hid_channels, out_channels),
                                 nn.BatchNorm1D(out_channels))

        init_backbone_weight_simclr(self.mlp)

    def init_parameters(self, init_linear='normal'):
        _init_parameters(self, init_linear)

    def forward(self, x):
        x = layers.squeeze(x, axes=[])
        hidden = self.mlp(x)
        hidden = layers.l2_normalize(hidden, -1)
        return hidden


@NECKS.register()
class SwAVNeck(nn.Layer):
    """The non-linear neck in SwAV: fc-bn-relu-fc-normalization.
    """
    def __init__(self,
                 in_channels,
                 hid_channels,
                 out_channels,
                 with_l2norm=True,
                 with_avg_pool=True):
        super(SwAVNeck, self).__init__()

        self.with_l2norm = with_l2norm
        self.with_avg_pool = with_avg_pool
        if with_avg_pool:
            self.avgpool = nn.AdaptiveAvgPool2D((1, 1))
        if out_channels == 0:
            self.projection_neck = None
        elif hid_channels == 0:
            self.projection_neck = nn.Linear(in_channels, out_channels)
        else:
            self.projection_neck = nn.Sequential(
                nn.Linear(in_channels, hid_channels),
                nn.BatchNorm1D(hid_channels), nn.ReLU(),
                nn.Linear(hid_channels, out_channels)
            )

    def forward_projection(self, x):
        if self.projection_neck is not None:
            x = self.projection_neck(x)
        if self.with_l2norm:
            x = nn.functional.normalize(x, axis=1, p=2)
        return x

    def forward(self, x):
        # forward computing
        # x: list of feature maps, len(x) according to len(num_crops)
        avg_out = []
        for _x in x:
            if self.with_avg_pool:
                _out = self.avgpool(_x)
                avg_out.append(_out)
        feat_vec = paddle.concat(avg_out)  # [sum(num_crops) * N, C]
        feat_vec = feat_vec.reshape([feat_vec.shape[0], -1])
        output = self.forward_projection(feat_vec)
        return output



@NECKS.register()
class MLP2d(nn.Layer):
    """The non-linear neck in pixpro.
    """

    def __init__(self, in_channels, hid_channels=4096, out_channels=256):
        super(MLP2d, self).__init__()

        self.linear1 = nn.Conv2D(in_channels,
                                 hid_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0,
                                 bias_attr=True)
        self.bn1 = nn.BatchNorm2D(hid_channels)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Conv2D(hid_channels,
                                 out_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0,
                                 bias_attr=True)
        self.init_parameters()

    def init_parameters(self, init_linear='kaiming'):
        _init_parameters(self, init_linear)
        return

    def forward(self, x):
        x = self.linear1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.linear2(x)

        return x
