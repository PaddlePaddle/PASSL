# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
import functools
import paddle.nn as nn

from .resnet import ResNet, BottleneckBlock

def kaiming_normal_init(param, **kwargs):
    initializer = nn.initializer.KaimingNormal(**kwargs)
    initializer(param, param.block)

def constant_init(param, **kwargs):
    initializer = nn.initializer.Constant(**kwargs)
    initializer(param, param.block)


class SwAVResNet(paddle.nn.Layer):
    def __init__(self, block, depth,
        normalize=False, output_dim=0, hidden_mlp=0,
        nmb_prototypes=0, eval_mode=False):

        super(SwAVResNet, self).__init__()
        self.l2norm = normalize
        self.eval_mode = eval_mode
        num_out_filters = 512

        self.avgpool = paddle.nn.AdaptiveAvgPool2D(output_size=(1, 1))

        if output_dim == 0:
            self.projection_head = None
        elif hidden_mlp == 0:
            self.projection_head = paddle.nn.Linear(in_features=
                num_out_filters * block.expansion, out_features=output_dim)
        else:
            self.projection_head = paddle.nn.Sequential(paddle.nn.Linear(
                in_features=num_out_filters * block.expansion, out_features
                =hidden_mlp), paddle.nn.BatchNorm1D(num_features=hidden_mlp,
                momentum=1 - 0.1, epsilon=1e-05, weight_attr=None,
                bias_attr=None, use_global_stats=True), paddle.nn.ReLU(),
                paddle.nn.Linear(in_features=hidden_mlp, out_features=
                output_dim))

        self.prototypes = None
        if isinstance(nmb_prototypes, list):
            self.prototypes = MultiPrototypes(output_dim, nmb_prototypes)
        elif nmb_prototypes > 0:
            self.prototypes = paddle.nn.Linear(in_features=output_dim,
                out_features=nmb_prototypes, bias_attr=False)
            for sublayer in self.sublayers():
                if isinstance(sublayer, nn.Conv2D):
                    kaiming_normal_init(sublayer.weight) # todo mode='fan_out',
                elif isinstance(sublayer, (nn.BatchNorm2D, nn.GroupNorm)):
                    constant_init(sublayer.weight, value=1.0)
                    constant_init(sublayer.bias, value=0.0)

        self.encoder = functools.partial(ResNet, block=block, depth=depth)(with_pool=False, class_num=0)

    def forward_backbone(self, x):
        x = self.encoder(x)

        if self.eval_mode:
            return x

        x = self.avgpool(x)
        x = paddle.flatten(x=x, start_axis=1)
        return x

    def forward_head(self, x):
        if self.projection_head is not None:
            x = self.projection_head(x)
        if self.l2norm:
            x = paddle.nn.functional.normalize(x=x, axis=1, p=2)
        if self.prototypes is not None:
            return x, self.prototypes(x)
        return x

    def forward(self, inputs):
        if not isinstance(inputs, list):
            inputs = [inputs]

        idx_crops = paddle.cumsum(x=paddle.unique_consecutive(x=paddle.
            to_tensor(data=[inp.shape[-1] for inp in inputs]),
            return_counts=True)[1], axis=0) # padiff
        start_idx = 0
        for end_idx in idx_crops:
            _out = self.forward_backbone(paddle.concat(x=inputs[start_idx:end_idx]))
            if start_idx == 0:
                output = _out
            else:
                output = paddle.concat(x=(output, _out))
            start_idx = end_idx
        return self.forward_head(output)


class MultiPrototypes(paddle.nn.Layer):
    def __init__(self, output_dim, nmb_prototypes):
        super(MultiPrototypes, self).__init__()
        self.nmb_heads = len(nmb_prototypes)
        for i, k in enumerate(nmb_prototypes):
            self.add_module('prototypes' + str(i), paddle.nn.Linear(
                in_features=output_dim, out_features=k, bias_attr=False))

    def forward(self, x):
        out = []
        for i in range(self.nmb_heads):
            out.append(getattr(self, 'prototypes' + str(i))(x))
        return out


def swavresnet50(**kwargs):
    return SwAVResNet(block=BottleneckBlock, depth=50, **kwargs)
