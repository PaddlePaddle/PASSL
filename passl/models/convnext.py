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
import os

import paddle
import paddle.nn as nn
from paddle.nn.functional import layer_norm

from passl.models.base_model import Model
from passl.models.vision_transformer import DropPath
from passl.nn import init
"""
ConvNext by paddle.
References: https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py
"""

__all__ = [
    "ConvNeXt", "convnext_tiny", "convnext_small", "convnext_base",
    "convnext_large", "convnext_xlarge"
]


class LayerNorm(nn.Layer):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6,
                 data_format="channels_last"):
        super().__init__()
        self.normalized_shape = (normalized_shape, )
        self.weight = self.create_parameter(
            shape=self.normalized_shape,
            default_initializer=paddle.nn.initializer.Constant(value=1.))
        self.bias = self.create_parameter(
            shape=self.normalized_shape,
            default_initializer=paddle.nn.initializer.Constant(value=0.))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError

    def forward(self, x):
        if self.data_format == "channels_last":
            return layer_norm(x, self.normalized_shape, self.weight, self.bias,
                              self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / paddle.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class Block(nn.Layer):
    """
    ConvNext Block.

    """

    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2D(
            dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)  # channel last
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = None
        if layer_scale_init_value > 0:
            self.gamma = self.create_parameter(
                shape=(dim, ),
                default_initializer=paddle.nn.initializer.Constant(
                    layer_scale_init_value))
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.transpose(perm=[0, 2, 3, 1])
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.transpose(perm=[0, 3, 1, 2])
        x = input + self.drop_path(x)
        return x


class ConvNeXt(Model):
    """ConvNeXt
        A Paddle impl of : `A ConvNet for the 2020s` - https://arxiv.org/pdf/2201.03545.pdf
    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """

    def __init__(
            self,
            in_chans=3,
            num_classes=1000,
            depths=[3, 3, 9, 3],
            dims=[96, 192, 384, 768],
            drop_path_rate=0.,
            layer_scale_init_value=1e-6,
            head_init_scale=1., ):
        super().__init__()
        self.downsample_layers = nn.LayerList()
        stem = nn.Sequential(
            nn.Conv2D(
                in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(
                dims[0], eps=1e-6, data_format="channels_first"), )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsampler_layer = nn.Sequential(
                LayerNorm(
                    dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2D(
                    dims[i], dims[i + 1], kernel_size=2, stride=2), )
            self.downsample_layers.append(downsampler_layer)

        self.stages = nn.LayerList()
        dp_rates = [
            x.item() for x in paddle.linspace(0, drop_path_rate, sum(depths))
        ]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(* [
                Block(
                    dim=dims[i],
                    drop_path=dp_rates[cur + j],
                    layer_scale_init_value=layer_scale_init_value)
                for j in range(depths[i])
            ])
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], epsilon=1e-6)
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)
        self.head.weight.set_value(self.head.weight * head_init_scale)
        self.head.bias.set_value(self.head.bias * head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2D, nn.Linear)):
            init.trunc_normal_(m.weight, std=.02)
            init.constant_(m.bias, 0)

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x.mean(
            [-2, -1]))  # global average pooling, (N, C, H, W) -> (N, C)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

    def load_pretrained(self, path, rank=0, finetune=False):
        # load pretrained model
        if not os.path.exists(path + '.pdparams'):
            raise ValueError("Model pretrain path {} does not "
                             "exists.".format(path))

        state_dict = self.state_dict()
        param_state_dict = paddle.load(path + ".pdparams")
        # for FP16 saving pretrained weight
        for key, value in param_state_dict.items():
            if key in param_state_dict and key in state_dict and param_state_dict[
                    key].dtype != state_dict[key].dtype:
                param_state_dict[key] = param_state_dict[key].astype(
                    state_dict[key].dtype)
        if not finetune:
            self.set_dict(param_state_dict)

    def save(self, path, local_rank=0, rank=0):
        if local_rank == 0:
            paddle.save(self.state_dict(), path + ".pdparams")


def convnext_tiny(**kwargs):
    model = ConvNeXt(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
    return model


def convnext_small(**kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], **kwargs)
    return model


def convnext_base(**kwargs):
    model = ConvNeXt(
        depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
    return model


def convnext_large(**kwargs):
    model = ConvNeXt(
        depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)
    return model


def convnext_xlarge(**kwargs):
    model = ConvNeXt(
        depths=[3, 3, 27, 3], dims=[256, 512, 1024, 2048], **kwargs)
    return model
