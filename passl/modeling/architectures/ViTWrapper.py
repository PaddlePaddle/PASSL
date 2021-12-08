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
from .builder import MODELS


@MODELS.register()
class ViTWrapper(nn.Layer):
    def __init__(self,
                 architecture=None,
                 label_smoothing=None
                 ):
        """A wrapper for a ViT model as specified in the paper.

        Args:
            architecture (dict): A dictionary containing the ViT instantiation parameters.
        """
        super().__init__()

        self.backbone = build_backbone(architecture)
        self.automatic_optimization = False
        self.label_smoothing = label_smoothing
        if self.label_smoothing is not None:
            assert self.label_smoothing > 0 and self.label_smoothing < 1
        
    def loss(self, x, label):
        losses = dict()
        
        class_num = x.shape[-1]
        if len(label.shape) == 1 or label.shape[-1] != class_num:
            label = F.one_hot(label, class_num)
            label = paddle.reshape(label, shape=[-1, class_num])
        if self.label_smoothing is not None:
            # vit style label smoothing
            with paddle.no_grad(): 
                label = label * (1.0 - self.label_smoothing) + self.label_smoothing
                
        loss = F.binary_cross_entropy_with_logits(x, label, reduction='none')
        loss = paddle.sum(loss, axis=-1)
        loss = loss.mean()

        losses['loss'] = loss
        losses['acc1'], losses['acc5'] = accuracy(x, label, topk=(1, 5))
        return losses

    def train_iter(self, *inputs, **kwargs):
        img, label = inputs
        mixup_fn = kwargs['mixup_fn']
        if mixup_fn is not None:
            img, label = mixup_fn(img, label)

        outs = self.backbone(img)
        outputs = self.loss(outs, label)
        return outputs

    def test_iter(self, *inputs, **kwargs):
        with paddle.no_grad():
            img, label = inputs
            outs = self.backbone(img)

        return outs

    def forward(self, *inputs, mode='train', **kwargs):
        if mode == 'train':
            return self.train_iter(*inputs, **kwargs)
        elif mode == 'test':
            return self.test_iter(*inputs, **kwargs)
        elif mode == 'extract':
            return self.backbone.forward_features(x)
        else:
            raise Exception("No such mode: {}".format(mode))
        
def accuracy(output, target, topk=(1, )):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with paddle.no_grad():
        maxk = max(topk)
        if target.dim() > 1:
            target = target.argmax(axis=-1)
        batch_size = target.shape[0]

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = paddle.cast(pred == target.reshape([1, -1]).expand_as(pred),
                              'float32')

        res = []
        for k in topk:
            correct_k = correct[:k].reshape([-1]).sum(0, keepdim=True)
            res.append(correct_k * 100.0 / batch_size)
        return res
