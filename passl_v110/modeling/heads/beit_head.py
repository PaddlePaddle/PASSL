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

import sys
import math
import paddle
import paddle.nn as nn
from paddle import multiply
from paddle.nn import Identity
import paddle.nn.functional as F

from .builder import HEADS

trunc_normal_ = nn.initializer.TruncatedNormal(std=0.02)
zeros_ = nn.initializer.Constant(value=0.0)
ones_ = nn.initializer.Constant(value=1.0)


@HEADS.register()
class BEiTClsHead(nn.Layer):
    """BEiT classifier head.

    Args:
        in_channels (int): Number of channels in the input feature map.
        num_classes (int): Number of categories excluding the background category.
    """
    def __init__(self, in_channels=2048, num_classes=1000, init_scale=0.001):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.criterion = nn.CrossEntropyLoss()
        self.fc_cls = nn.Linear(in_channels, num_classes)

        if isinstance(self.fc_cls, nn.Linear):
            trunc_normal_(self.fc_cls.weight)
            self.fc_cls.weight.set_value(
                self.fc_cls.weight.multiply(paddle.to_tensor(init_scale)))
            self.fc_cls.bias.set_value(
                self.fc_cls.bias.multiply(paddle.to_tensor(init_scale)))

    def forward(self, x):
        cls_score = self.fc_cls(x)
        return cls_score

    def loss(self, cls_score, labels):
        losses = dict()

        losses["loss"] = self.criterion(cls_score, labels)
        losses["acc1"], losses["acc5"] = accuracy(cls_score,
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
                              "float32")

        res = []
        for k in topk:
            correct_k = correct[:k].reshape([-1]).sum(0, keepdim=True)
            res.append(correct_k * 100.0 / batch_size)
        return res


@HEADS.register()
class BEiTPTHead(nn.Layer):
    """BEiT Pretrain Head.

    Args:
        in_channels (int): Number of channels in the input feature map.
        num_classes (int): Number of categories excluding the background category.
    """
    def __init__(self, in_channels=None, num_classes=None, init_scale=0.001):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, cls_score, labels):
        losses = dict()
        losses["loss"] = self.criterion(cls_score, labels)
        loss_value = losses["loss"].item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)
        #lossmlm_acc = (cls_score.max(-1) == labels).astype('float32').mean().item()

        losses["mlm_acc"] = accuracy(cls_score, labels)[0]
        return losses


def accuracy(output, target, topk=(1, )):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with paddle.no_grad():
        maxk = max(topk)
        batch_size = target.shape[0]

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = paddle.cast(pred == target.reshape([1, -1]).expand_as(pred),
                              "float32")

        res = []
        for k in topk:
            correct_k = correct[:k].reshape([-1]).sum(0, keepdim=True)
            res.append(correct_k * 100.0 / batch_size)
        return res


@HEADS.register()
class BEiTFTHead(nn.Layer):
    """BEiT Finetune Head.

    Args:
        in_channels (int): Number of channels in the input feature map.
        num_classes (int): Number of categories excluding the background category.
    """
    def __init__(self, in_channels=None, num_classes=None, init_scale=0.001):
        super(BEiTFTHead, self).__init__()
        self.head = nn.Linear(in_channels,
                              num_classes) if num_classes > 0 else Identity()
        self.criterion = nn.CrossEntropyLoss()
        trunc_normal_(self.head.weight)
        self.apply(self._init_weights)

        self.head.weight.set_value(
            multiply(self.head.weight, paddle.to_tensor(init_scale)))
        self.head.bias.set_value(
            multiply(self.head.bias, paddle.to_tensor(init_scale)))

    def forward(self, x):
        x = self.head(x)
        return x

    def loss(self, x, labels, soft=True):
        losses = dict()
        if soft:
            losses['loss'] = paddle.sum(-labels * F.log_softmax(x, axis=-1),
                                        axis=-1).mean()
        else:
            losses["loss"] = self.criterion(x, labels)
        losses['acc1'], losses['acc5'] = accuracy(x, labels, topk=(1, 5))
        return losses

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            zeros_(m.bias)
            ones_(m.weight)


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
