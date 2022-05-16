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


@HEADS.register()
class SimSiamContrastiveHead(nn.Layer):
    """Head for simsiam contrastive learning."""
    def __init__(self):
        super(SimSiamContrastiveHead, self).__init__()
        self.criterion = nn.CosineSimilarity(axis=1)

    def forward(self, p1, p2, z1, z2):
        """Forward head.

        Args:
            p1 (Tensor): output of predictor1.
            p2 (Tensor): output of predictor2.
            z1 (Tensor): output of encoder1.
            z2 (Tensor): output of encoder2.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        outputs = dict()
        outputs['loss'] = -(self.criterion(p1, z2).mean() + self.criterion(p2, z1).mean()) * 0.5
        return outputs
