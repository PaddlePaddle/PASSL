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
""" MAE in PaddlePaddle

Paper: 'MAE: Masked Autoencoders Are Scalable Vision Learners'
    - https://arxiv.org/abs/2111.06377

"""

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from ..backbones import build_backbone
from ..heads import build_head
from .builder import MODELS


@MODELS.register()
class MAE_PRETRAIN(nn.Layer):
    def __init__(self, architecture=None):
        """A wrapper for a ViT model as specified in the paper.

        Args:
            architecture (dict): A dictionary containing the MAE instantiation parameters.
        """
        super().__init__()

        self.backbone = build_backbone(architecture)

    def train_iter(self, *inputs, **kwargs):
        img = inputs
        loss, pred, mask = self.backbone(img)
        return loss, pred, mask

    def forward(self, *inputs, mode='train', **kwargs):
        if mode == 'train':
            return self.train_iter(*inputs, **kwargs)
        elif mode == 'test':
            return self.test_iter(*inputs, **kwargs)
        elif mode == 'extract':
            return self.backbone(*inputs)
        else:
            raise Exception("No such mode: {}".format(mode))


@MODELS.register()
class MAE_FINETUNE(nn.Layer):
    def __init__(self, architecture=None, head=None):
        """A wrapper for a ViT model as specified in the paper.

        Args:
            architecture (dict): A dictionary containing the MAE instantiation parameters.
        """
        super().__init__()

        self.backbone = build_backbone(architecture)
        self.head = build_head(head)

    def backbone_forward(self, x):
        x = self.backbone(x)
        return x

    def train_iter(self, *inputs, **kwargs):
        img, label = inputs
        cls_token = self.backbone_forward(img)
        outs = self.head(cls_token)
        loss_inputs = (outs, label)
        outputs = self.head.loss(*loss_inputs)
        return outputs

    def forward(self, *inputs, mode='train', **kwargs):
        if mode == 'train':
            return self.train_iter(*inputs, **kwargs)
        elif mode == 'test':
            return self.test_iter(*inputs, **kwargs)
        elif mode == 'extract':
            return self.backbone(*inputs)
        else:
            raise Exception("No such mode: {}".format(mode))
