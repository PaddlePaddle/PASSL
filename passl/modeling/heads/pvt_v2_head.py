# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserve.
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
import paddle.nn.functional as F

from ...modules.init import reset_parameters, trunc_normal_, constant_
from .builder import HEADS
from .clas_head import ClasHead


@HEADS.register()
class PyramidVisionTransformerV2ClsHead(ClasHead):
    """Pyramid Vision Transformer V2 classifier head.

    Args:
        with_avg_pool (bool): Use average pooling or not. Default: False.
        in_channels (int): Number of channels in the input feature map.
        num_classes (int): Number of categories excluding the background
            category.
    """
    def __init__(self, with_avg_pool=False, in_channels=2048, num_classes=1000):
        super(PyramidVisionTransformerV2ClsHead, self).__init__()
        self.with_avg_pool = with_avg_pool
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.fc_cls = nn.Linear(in_channels, num_classes)
        if self.with_avg_pool:
            self.avg_pool = nn.AdaptiveAvgPool1D(1)

        trunc_normal_(self.fc_cls.weight, mean=0.0, std=0.02)
        constant_(self.fc_cls.bias, 0)

    def forward(self, x):
        if self.with_avg_pool:
            x = self.avg_pool(x)
        x = paddle.flatten(x, 1)
        cls_score = self.fc_cls(x)
        return cls_score

    def loss(self, x, labels):
        losses = dict()
        losses['loss'] = paddle.sum(-labels * F.log_softmax(x, axis=-1),
                                    axis=-1).mean()
        losses['acc1'], losses['acc5'] = accuracy(x, labels, topk=(1, 5))
        return losses


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
