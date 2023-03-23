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
import numpy as np

from ...modules.init import init_backbone_weight
from ...modules import freeze_batchnorm_statictis
from .builder import MODELS
from ..backbones import build_backbone
from ..necks import build_neck
from ..heads import build_head


def regression_loss(q, k, coord_q, coord_k, pos_ratio=0.5):
    """ q, k: N * C * H * W
        coord_q, coord_k: N * 4 (x_upper_left, y_upper_left, x_lower_right, y_lower_right)
    """
    N, C, H, W = q.shape
    # [bs, feat_dim, 49]
    q = q.reshape([N, C, -1])
    k = k.reshape([N, C, -1])

    # generate center_coord, width, height
    # [1, 7, 7]
    x_array = paddle.arange(0.,
                            float(W),
                            dtype=coord_q.dtype,
                            device=coord_q.device).reshape([1, 1, -1
                                                            ]).repeat(1, H, 1)
    y_array = paddle.arange(0.,
                            float(H),
                            dtype=coord_q.dtype,
                            device=coord_q.device).reshape([1, -1,
                                                            1]).repeat(1, 1, W)
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
    q_bin_diag = paddle.sqrt(q_bin_width**2 + q_bin_height**2)
    k_bin_diag = paddle.sqrt(k_bin_width**2 + k_bin_height**2)
    max_bin_diag = paddle.max(q_bin_diag, k_bin_diag)

    # [bs, 7, 7]
    center_q_x = (x_array + 0.5) * q_bin_width + q_start_x
    center_q_y = (y_array + 0.5) * q_bin_height + q_start_y
    center_k_x = (x_array + 0.5) * k_bin_width + k_start_x
    center_k_y = (y_array + 0.5) * k_bin_height + k_start_y

    # [bs, 49, 49]
    dist_center = paddle.sqrt(
        (center_q_x.reshape([-1, H * W, 1]) -
         center_k_x.reshape([-1, 1, H * W]))**2 +
        (center_q_y.reshape([-1, H * W, 1]) -
         center_k_y.reshape([-1, 1, H * W]))**2) / max_bin_diag
    pos_mask = (dist_center < pos_ratio).float().detach()

    # [bs, 49, 49]
    logit = paddle.bmm(q.transpose([0, 2, 1]), k)

    loss = (logit * pos_mask).sum(-1).sum(-1) / (pos_mask.sum(-1).sum(-1) +
                                                 1e-6)

    return -2 * loss.mean()


@MODELS.register()
class PixPro(nn.Layer):

    def __init__(self,
                 backbone,
                 neck=None,
                 predictor=None,
                 head=None,
                 pixpro_p=2,
                 pixpro_momentum=0.99,
                 pixpro_clamp_value=0.,
                 pixpro_transform_layer=1,
                 pixpro_ins_loss_weight=1,
                 use_synch_bn=True):
        super(PixPro, self).__init__()

        # parse arguments
        self.pixpro_p = pixpro_p
        self.pixpro_momentum = pixpro_momentum
        self.pixpro_clamp_value = pixpro_clamp_value
        self.pixpro_transform_layer = pixpro_transform_layer
        self.pixpro_ins_loss_weight = pixpro_ins_loss_weight

        # create the encoder
        self.backbone = build_backbone(backbone)
        self.neck = build_neck(neck)
        self.head = build_head(head)

        # create the encoder_k
        self.backbone_k = build_backbone(backbone)
        self.neck_k = build_neck(neck)

        for param_q, param_k in zip(self.backbone.parameters(),
                                    self.backbone_k.parameters()):
            param_k.set_value(param_q)  # initialize
            param_k.stop_gradient = True  # not update by gradient

        for param_q, param_k in zip(self.neck.parameters(),
                                    self.neck_k.parameters()):
            param_k.set_value(param_q)  # initialize
            param_k.stop_gradient = True  # not update by gradient

        if use_synch_bn:
            self.backbone = nn.SyncBatchNorm.convert_sync_batchnorm(
                self.backbone)
            self.backbone_k = nn.SyncBatchNorm.convert_sync_batchnorm(
                self.backbone_k)
            self.neck = nn.SyncBatchNorm.convert_sync_batchnorm(self.neck)
            self.neck_k = nn.SyncBatchNorm.convert_sync_batchnorm(self.neck_k)

        if self.pixpro_transform_layer == 0:
            self.value_transform = paddle.nn.Identity()
        elif self.pixpro_transform_layer == 1:
            self.value_transform = nn.Conv2D(
                256,
                256,
                kernel_size=1,
                stride=1,
                padding=0,
                bias_attr=True,
                weight_attr=nn.initializer.KaimingUniform())
        elif self.pixpro_transform_layer == 2:
            self.value_transform = MLP2d(in_dim=256, inner_dim=256, out_dim=256)
        else:
            raise NotImplementedError

        if self.pixpro_ins_loss_weight > 0.:
            self.neck_instance = build_neck(neck)
            self.neck_instance_k = build_neck(neck)
            self.predictor = build_neck(predictor)

            for param_q, param_k in zip(self.neck_instance.parameters(),
                                        self.neck_instance_k.parameters()):
                param_k.set_value(param_q)
                param_k.stop_gradient = True

            if use_synch_bn:
                self.neck_instance = nn.SyncBatchNorm.convert_sync_batchnorm(
                    self.neck_instance)
                self.neck_instance_k = nn.SyncBatchNorm.convert_sync_batchnorm(
                    self.neck_instance_k)
                self.predictor = nn.SyncBatchNorm.convert_sync_batchnorm(
                    self.predictor)

            self.avgpool = nn.AvgPool2D(7, stride=1)

    @paddle.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        _contrast_momentum = 1. - (1. - self.pixpro_momentum) * (
            np.cos(np.pi * self.k / self.K) + 1) / 2.
        for param_q, param_k in zip(self.backbone.parameters(),
                                    self.backbone_k.parameters()):
            paddle.assign((param_k * _contrast_momentum + param_q *
                           (1. - _contrast_momentum)), param_k)
            param_k.stop_gradient = True

        for param_q, param_k in zip(self.neck.parameters(),
                                    self.neck_k.parameters()):
            paddle.assign((param_k * _contrast_momentum + param_q *
                           (1. - _contrast_momentum)), param_k)
            param_k.stop_gradient = True

        if self.pixpro_ins_loss_weight > 0.:
            for param_q, param_k in zip(self.neck_instance.parameters(),
                                        self.neck_instance_k.parameters()):
                paddle.assign((param_k * _contrast_momentum + param_q *
                               (1. - _contrast_momentum)), param_k)
                param_k.stop_gradient = True

    def featprop(self, feat):
        N, C, H, W = feat.shape
        # Value transformation
        feat_value = self.value_transform(feat)
        feat_value = nn.functional.normalize(feat_value, axis=1)
        feat_value = feat_value.reshape([N, C, -1])
        # Similarity calculation
        feat = nn.functional.normalize(feat, axis=1)
        # [N, C, H * W]
        feat = feat.reshape([N, C, -1])

        # [N, H * W, H * W]
        attention = paddle.bmm(feat.transpose([0, 2, 1]), feat)
        attention = paddle.clip(attention, min=self.pixpro_clamp_value)
        if self.pixpro_p < 1.:
            attention = attention + 1e-6
        attention = attention**self.pixpro_p

        # [N, C, H * W]
        feat = paddle.bmm(feat_value, attention.transpose([0, 2, 1]))

        return feat.reshape([N, C, H, W])

    def train_iter(self, im_1, im_2, coord1, coord2, **kwargs):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """
        self.k = kwargs['current_iter']
        self.K = kwargs['total_iters']

        # compute query features
        feat_1 = self.backbone(im_1)  # queries: NxC
        feat_2 = self.backbone(im_2)

        pred_1 = self.neck(feat_1)
        pred_2 = self.neck(feat_2)

        pred_1 = self.featprop(pred_1)
        pred_1 = nn.functional.normalize(pred_1, axis=1)
        pred_2 = self.featprop(pred_2)
        pred_2 = nn.functional.normalize(pred_2, axis=1)

        if self.pixpro_ins_loss_weight > 0.:
            proj_instance_1 = self.neck_instance(feat_1)
            proj_instance_2 = self.neck_instance(feat_2)
            pred_instance_1 = self.predictor(proj_instance_1)
            pred_instance_1 = nn.functional.normalize(
                self.avgpool(pred_instance_1).reshape(
                    [pred_instance_1.shape[0], -1]),
                axis=1)

            pred_instance_2 = self.predictor(proj_instance_2)
            pred_instance_2 = nn.functional.normalize(
                self.avgpool(pred_instance_2).reshape(
                    [pred_instance_2.shape[0], -1]),
                axis=1)

        # compute key features
        with paddle.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            feat_1_ng = self.backbone_k(im_1)  # keys: NxC
            proj_1_ng = self.neck_k(feat_1_ng)
            feat_2_ng = self.backbone_k(im_2)
            proj_2_ng = self.neck_k(feat_2_ng)

            proj_1_ng = nn.functional.normalize(proj_1_ng, axis=1)
            proj_2_ng = nn.functional.normalize(proj_2_ng, axis=1)

            if self.pixpro_ins_loss_weight > 0.:
                proj_instance_1_ng = self.neck_instance_k(feat_1_ng)
                proj_instance_2_ng = self.neck_instance_k(feat_2_ng)

                proj_instance_1_ng = nn.functional.normalize(
                    self.avgpool(proj_instance_1_ng).reshape(
                        [proj_instance_1_ng.shape[0], -1]),
                    axis=1)
                proj_instance_2_ng = nn.functional.normalize(
                    self.avgpool(proj_instance_2_ng).reshape(
                        [proj_instance_2_ng.shape[0], -1]),
                    axis=1)
        q = [pred_1, proj_2_ng, coord1, coord2]
        k = [pred_2, proj_1_ng, coord2, coord1]

        if self.pixpro_ins_loss_weight > 0.:
            q.append(pred_instance_1)
            q.append(proj_instance_2_ng)
            k.append(pred_instance_2)
            k.append(proj_instance_1_ng)
        outputs = self.head(q, k, self.pixpro_ins_loss_weight)
        return outputs

    def forward(self, *inputs, mode='train', **kwargs):
        if mode == 'train':
            return self.train_iter(*inputs, **kwargs)
        elif mode == 'test':
            return self.test_iter(*inputs, **kwargs)
        elif mode == 'extract':
            return self.backbone(*inputs)
        else:
            raise Exception("No such mode: {}".format(mode))
