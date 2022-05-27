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

from ...modules.init import init_backbone_weight
from .builder import MODELS
from ..backbones import build_backbone
from ..necks import build_neck
from ..heads import build_head


@MODELS.register()
class Classification(nn.Layer):
    """
    Simple image classification.
    """

    def __init__(self, backbone, with_sobel=False, head=None):
        super(Classification, self).__init__()

        self.with_sobel = with_sobel
        if with_sobel:
            # TODO: add Sobel
            pass
        self.backbone = build_backbone(backbone)
        if head is not None:
            self.head = build_head(head)

    def backbone_forward(self, x):
        x = self.backbone(x)
        return x

    def train_iter(self, *inputs, **kwargs):
        img, label = inputs
        x = self.backbone_forward(img)
        outs = self.head(x)
        loss_inputs = (outs, label)
        outputs = self.head.loss(*loss_inputs)
        return outputs

    def test_iter(self, *inputs, **kwargs):
        with paddle.no_grad():
            img, label = inputs
            x = self.backbone_forward(img)
            outs = self.head(x)

        return outs

    def infer_iter(self, *inputs, **kwargs):
        with paddle.no_grad():
            x = self.backbone_forward(*inputs)
            outs = self.head(x)

        return outs

    def forward(self, *inputs, mode='train', **kwargs):
        if mode == 'train':
            return self.train_iter(*inputs, **kwargs)
        elif mode == 'test':
            return self.test_iter(*inputs, **kwargs)
        elif mode == 'extract':
            return self.backbone(*inputs)
        elif mode == 'infer':
            return self.infer_iter(*inputs)
        else:
            raise Exception("No such mode: {}".format(mode))
