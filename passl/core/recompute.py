# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import functools

import paddle
from paddle import nn
from paddle.distributed.fleet.utils import recompute


def wrap_forward(func, newfunc):
    @functools.wraps(func)
    def run(*args, **kwargs):
        return newfunc(func, *args, **kwargs)

    return run


def recompute_forward(func, *args, **kwargs):
    return recompute(func, *args, **kwargs)


def recompute_warp(model, layerlist_interval=1, names=[]):

    for name, layer in model._sub_layers.items():
        if isinstance(layer, nn.LayerList):
            for idx, sub_layer in enumerate(layer):
                if layerlist_interval >= 1 and idx % layerlist_interval == 0:
                    sub_layer.forward = wrap_forward(sub_layer.forward,
                                                     recompute_forward)
        if name in names:
            layer.forward = wrap_forward(layer.forward, recompute_forward)
