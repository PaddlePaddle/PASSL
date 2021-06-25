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
import paddle.nn.functional as F
import paddle.fluid.layers as layers


LARGE_NUM = 1e9

@MODELS.register()
class SimCLR(nn.Layer):
    """
    Simple image SimCLR.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 head=None,
                 dim=128,
                 T=0.5):
        super(SimCLR, self).__init__()
        self.T = T

        self.encoder = nn.Sequential(build_backbone(backbone),
                                       build_neck(neck))
        
        self.backbone = self.encoder[0]
        self.head = build_head(head)



    def train_iter(self, *inputs, **kwargs):
        img_q, img_k = inputs
        img_con = [img_q, img_k]
        img_con = paddle.concat(img_con)
        con = self.encoder(img_con)
        con = layers.l2_normalize(con, -1)
        q, k = layers.split(con, num_or_sections=2, dim=0)
        outputs = self.head(q, k)
      
        return outputs
    def test_iter(self, *inputs, **kwargs):
        with paddle.no_grad():
            img, label = inputs
            x = self.backbone_forward(img)
            outs = self.head(x)

        return outs

    def forward(self, *inputs, mode='train', **kwargs):
        if mode == 'train':
            return self.train_iter(*inputs, **kwargs)
        elif mode == 'test':
            return self.test_iter(*inputs, **kwargs)
        elif mode == 'extract':
            return self.backbone(*inputs)
        else:
            raise Exception("No such mode: {}".format(mode))



