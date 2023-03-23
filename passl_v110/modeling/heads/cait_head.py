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

from .builder import HEADS

trunc_normal_ = nn.initializer.TruncatedNormal(std=0.02)
zeros_ = nn.initializer.Constant(value=0.0)


@HEADS.register()
class CaitClsHead(nn.Layer):
    """Vision Transformer classifier head.

    Args:
        with_avg_pool (bool): Use average pooling or not. Default: False.
        in_channels (int): Number of channels in the input feature map.
        num_classes (int): Number of categories excluding the background
            category.
    """
    def __init__(self, in_channels=2048, num_classes=1000):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.criterion = nn.CrossEntropyLoss()
        self.fc_cls = nn.Linear(in_channels, num_classes)

        if isinstance(self.fc_cls, nn.Linear):
            trunc_normal_(self.fc_cls.weight)
            if self.fc_cls.bias is not None:
                zeros_(self.fc_cls.bias)

    def forward(self, x):
        cls_score = self.fc_cls(x)
        return cls_score

    def loss(self, cls_score, labels):
        losses = dict()

        losses['loss'] = self.criterion(cls_score, labels)
        losses['acc1'], losses['acc5'] = accuracy(cls_score,
                                                  labels,
                                                  topk=(1, 5))
        return losses


def accuracy(output, target, topk=(1, )):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with paddle.no_grad():
        maxk = max(topk)
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
