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
import paddle.nn.functional as F

from .builder import HEADS


@HEADS.register()
class MBHead(nn.Layer):
    """Head for MoCo_BYOL.

    Args:
        temperature (float): The temperature hyper-parameter that
            controls the concentration level of the distribution.
            Default: 0.1.
    """

    def __init__(self):
        super(MBHead, self).__init__()
        self.criterion_q = nn.CrossEntropyLoss()
        self.criterion_k = nn.CrossEntropyLoss()

    def forward(self, preds, targets, logits_q, labels_q, logits_k, labels_k):
        bz = preds.shape[0]
        preds_norm = F.normalize(preds, axis=1)
        targets_norm = F.normalize(targets, axis=1)
        outputs = dict()
        outputs['loss_byol'] = 2 - 2 * (preds_norm * targets_norm).sum() / bz 
        outputs['loss_moco_q'] = self.criterion_q(logits_q, labels_q) 
        outputs['loss_moco_k'] = self.criterion_k(logits_k, labels_k) 
        outputs['loss'] = outputs['loss_byol'] + 0.03 * outputs['loss_moco_q'] + 0.03 * outputs['loss_moco_k']  # noqa
        return outputs