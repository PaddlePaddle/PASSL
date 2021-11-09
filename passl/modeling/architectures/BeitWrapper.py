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

import math
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import paddle.distributed as dist

from ..backbones import build_backbone
from ..heads import build_head
from .builder import MODELS


@MODELS.register()
class BeitWrapper(nn.Layer):
    def __init__(self, architecture=None, head=None):
        """A wrapper for a ViT model as specified in the paper.

        Args:
            architecture (dict): A dictionary containing the ViT instantiation parameters.
        """
        super().__init__()

        self.backbone = build_backbone(architecture)
        self.automatic_optimization = False
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

    def validation_step(self, val_batch, idx):
        image, text = val_batch
        image_logits, text_logits = self.forward(image, text)
        ground_truth = paddle.arange(len(image_logits))
        loss = (self.image_loss(image_logits, ground_truth) +
                self.text_loss(text_logits, ground_truth)).div(2)
        self.log('val_loss', loss)
