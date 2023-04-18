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

# Ref: https://github.com/facebookresearch/simsiam/blob/main/simsiam/builder.py

import paddle.nn as nn
from passl.nn import init

from .resnet import ResNet, BottleneckBlock

__all__ = [
    'SimSiamLinearProbe',
    'simsiam_resnet50_linearprobe',
]


class SimSiamLinearProbe(ResNet):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # freeze all layers but the last fc
        for name, param in self.named_parameters():
            if name not in ['fc.weight', 'fc.bias']:
                param.stop_gradient = True

        # optimize only the linear classifier
        parameters = list(
            filter(lambda p: not p.stop_gradient, self.parameters()))
        assert len(parameters) == 2  # weight, bias

        init.normal_(self.fc.weight, mean=0.0, std=0.01)
        init.zeros_(self.fc.bias)

        self.apply(self._freeze_norm)

    def _freeze_norm(self, layer):
        if isinstance(layer, (nn.layer.norm._BatchNormBase)):
            layer._use_global_stats = True


def simsiam_resnet50_linearprobe(**kwargs):
    model = SimSiamLinearProbe(block=BottleneckBlock, depth=50, **kwargs)
    return model
