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
from paddle.regularizer import L2Decay
from paddle import ParamAttr
import paddle.nn.functional as F

from .builder import HEADS


@HEADS.register()
class PixProHead(nn.Layer):
    def __init__(self, pos_ratio=0.5, return_accuracy=False):
        super(PixProHead, self).__init__()
        self.pos_ratio = pos_ratio
        self.return_accuracy = return_accuracy

    def regression_loss(self, q, k, coord_q, coord_k):
        N, C, H, W = q.shape
        # [bs, feat_dim, 49]
        q = q.reshape([N, C, -1])
        k = k.reshape([N, C, -1])

        # generate center_coord, width, height
        # [1, 7, 7]
        x_array = paddle.arange(0., float(W), dtype=coord_q.dtype).reshape([1, 1, -1]).tile([1, H, 1])
        y_array = paddle.arange(0., float(H), dtype=coord_q.dtype).reshape([1, -1, 1]).tile([1, 1, W])
        # [bs, 1, 1]
        q_bin_width = ((coord_q[:, 2] - coord_q[:, 0]) / W).reshape([-1, 1, 1])
        q_bin_height = ((coord_q[:, 3] - coord_q[:, 1]) / H).reshape([-1, 1, 1])
        k_bin_width = ((coord_k[:, 2] - coord_k[:, 0]) / W).reshape([-1, 1, 1])
        k_bin_height = ((coord_k[:, 3] - coord_k[:, 1]) / H).reshape([-1, 1, 1])
        # [bs, 1, 1]
        q_start_x = coord_q[:, 0].reshape([-1, 1, 1])
        q_start_y = coord_q[:, 1].reshape([-1, 1, 1])
        k_start_x = coord_k[:, 0].reshape([-1, 1, 1])
        k_start_y = coord_k[:, 1].reshape([-1, 1, 1])

        # [bs, 1, 1]
        q_bin_diag = paddle.sqrt(q_bin_width ** 2 + q_bin_height ** 2)
        k_bin_diag = paddle.sqrt(k_bin_width ** 2 + k_bin_height ** 2)
        max_bin_diag = paddle.maximum(q_bin_diag, k_bin_diag)
        # [bs, 7, 7]
        center_q_x = (x_array + 0.5) * q_bin_width + q_start_x
        center_q_y = (y_array + 0.5) * q_bin_height + q_start_y
        center_k_x = (x_array + 0.5) * k_bin_width + k_start_x
        center_k_y = (y_array + 0.5) * k_bin_height + k_start_y

        # [bs, 49, 49]
        dist_center = paddle.sqrt((center_q_x.reshape([-1, H * W, 1]) - center_k_x.reshape([-1, 1, H * W])) ** 2
                                + (center_q_y.reshape([-1, H * W, 1]) - center_k_y.reshape([-1, 1, H * W])) ** 2) / max_bin_diag
        pos_mask = paddle.cast(dist_center < self.pos_ratio, "float32").detach()
        # [bs, 49, 49]
        logit = paddle.bmm(q.transpose([0, 2, 1]), k)
        loss = (logit * pos_mask).sum(-1).sum(-1) / (pos_mask.sum(-1).sum(-1) + 1e-6)
        return -2 * loss.mean()
    
    def instance_regression_loss(self, x, y):
        return -2. * paddle.einsum('nc, nc->n', x, y).mean()

    def forward(self, q, k, pixpro_ins_loss_weight=0.):
        if pixpro_ins_loss_weight > 0.: 
            pred_1, proj_2_ng, coord1, coord2, pred_instance_1, proj_instance_2_ng = q
            pred_2, proj_1_ng, coord2, coord1, pred_instance_2, proj_instance_1_ng = k 
            loss_instance = self.instance_regression_loss(pred_instance_1, proj_instance_2_ng) + \
                         self.instance_regression_loss(pred_instance_2, proj_instance_1_ng)
            loss_instance = pixpro_ins_loss_weight * loss_instance
        else:
            pred_1, proj_2_ng, coord1, coord2 = q
            pred_2, proj_1_ng, coord2, coord1 = k 
            loss_instance = 0.
        
        loss = loss_instance + self.regression_loss(pred_1, proj_2_ng, coord1, coord2) \
            + self.regression_loss(pred_2, proj_1_ng, coord2, coord1)
        outputs = dict()
        outputs['loss'] = loss
        return outputs
