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

from .builder import MODELS
from ..backbones import build_backbone
from ..necks import build_neck
from ..heads import build_head


@MODELS.register()
class SwAV(nn.Layer):
    """
    Build a SwAV model with: a backbone, a neck and a head.
    https://arxiv.org/abs/2011.09157
    """
    def __init__(self,
                 backbone,
                 neck=None,
                 head=None,
                 use_synch_bn=True):
        super(SwAV, self).__init__()

        self.backbone = build_backbone(backbone)
        self.neck = build_neck(neck)
        self.head = build_head(head)

        # Convert BatchNorm*d to SyncBatchNorm*d
        if use_synch_bn:
            self.backbone = nn.SyncBatchNorm.convert_sync_batchnorm(self.backbone)
            self.neck = nn.SyncBatchNorm.convert_sync_batchnorm(self.neck)

    def train_iter(self, *inputs, **kwargs):
        assert isinstance(inputs, (list, tuple))

        # multi-res forward passes
        idx_crops = paddle.cumsum(paddle.unique_consecutive(
                paddle.to_tensor([inp.shape[-1] for inp in inputs]),
                return_counts=True
        )[1], 0)

        start_idx = 0
        output = []
        for end_idx in idx_crops:
            _out = self.backbone(paddle.concat(inputs[start_idx: end_idx]))
            output.append(_out)
            start_idx = end_idx
        output = self.neck(output)
        outputs = self.head(output)
        return outputs

    def forward(self, *inputs, mode='train', **kwargs):
        if mode == 'train':
            return self.train_iter(*inputs, **kwargs)
        else:
            raise Exception("No such mode: {}".format(mode))
