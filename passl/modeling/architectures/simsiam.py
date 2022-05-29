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
from ...modules import freeze_batchnorm_statictis
from .builder import MODELS
from ..backbones import build_backbone
from ..necks import build_neck
from ..heads import build_head


@MODELS.register()
class SimSiam(nn.Layer):
    """
    Build a SimSiam model.
    https://arxiv.org/abs/2011.10566
    """
    def __init__(self,
                 backbone,
                 head=None,
                 predictor=None,
                 dim=2048,
                 use_synch_bn=True
                ):
        """
        Args:
            backbone (dict): config of backbone.
            head (dict): config of head.
            predictor (dict): config of predictor.
            use_synch_bn (bool): whether apply apply sync bn. 
        """
        super(SimSiam, self).__init__()

        # Create the encoder
        # number classes is the output fc dimension, zero-initialize last BNs
        self.encoder = build_backbone(backbone)

        # build a 3-layer projector
        prev_dim = self.encoder.fc.weight.shape[0]
        self.encoder.fc = nn.Sequential(nn.Linear(prev_dim, prev_dim, bias_attr=False),
                                        nn.BatchNorm1D(prev_dim),
                                        nn.ReLU(),
                                        nn.Linear(prev_dim, prev_dim, bias_attr=False),
                                        nn.BatchNorm1D(prev_dim),
                                        nn.ReLU(),
                                        self.encoder.fc,
                                        nn.BatchNorm1D(dim, weight_attr=False, bias_attr=False))
        self.encoder.fc[6].bias.stop_gradient = True

        self.predictor = build_neck(predictor)
        self.head = build_head(head)

        # Convert BatchNorm*d to SyncBatchNorm*d
        if use_synch_bn:
            self.encoder = nn.SyncBatchNorm.convert_sync_batchnorm(self.encoder)
            self.predictor = nn.SyncBatchNorm.convert_sync_batchnorm(self.predictor)

    def train_iter(self, *inputs, **kwargs):
        x1, x2 = inputs

        # compute features for one view
        z1 = self.encoder(x1)  # NxC
        z2 = self.encoder(x2)  # NxC

        p1 = self.predictor(z1)  # NxC
        p2 = self.predictor(z2)  # NxC

        outputs = self.head(p1, p2, z1.detach(), z2.detach())
        return outputs

    def forward(self, *inputs, mode='train', **kwargs):
        if mode == 'train':
            return self.train_iter(*inputs, **kwargs)
        else:
            raise Exception("No such mode: {}".format(mode))
