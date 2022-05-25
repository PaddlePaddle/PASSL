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
from ...utils.misc import has_batchnorms


@MODELS.register()
class DINO(nn.Layer):
    """
    Build a DINO model with: a teacher and a student.
    """
    def __init__(self,
                 architecture=None,
                 neck=None,
                 head=None,
                 m=0.996,
                 epoch=0,
                 drop_path_rate=0.1,
                 norm_last_layer=True):
        """
        Args:
            backbone (dict): config of backbone.
            neck (dict): config of neck.
            head (dict): config of head.
            scale (list|tuple): Range of size of the origin size cropped. Default: (0.08, 1.0)
            dim (int): feature dimension. Default: 128.
            m (float): moco momentum of updating key encoder. Default: 0.999.
        """
        super(DINO, self).__init__()

        self.m = m
        self.epoch = epoch

        # create the teacher and student
        self.teacher = nn.Sequential(build_backbone(architecture),
                                     build_neck(neck))

        # add specific cfg to student
        architecture.update({'drop_path_rate': drop_path_rate})
        neck.update({'norm_last_layer': norm_last_layer})
        self.student = nn.Sequential(build_backbone(architecture),
                                     build_neck(neck))

        self.head = build_head(head)

        if has_batchnorms(self.student):
            self.teacher = nn.SyncBatchNorm.convert_sync_batchnorm(self.teacher)
            self.student = nn.SyncBatchNorm.convert_sync_batchnorm(self.student)

        self.teacher.set_state_dict(self.student.state_dict())
        for p in self.teacher.parameters():
            p.stop_gradient = True

    @paddle.no_grad()
    def _momentum_update_teacher(self):
        """
        Momentum update of the teacher
        """
        for param_q, param_k in zip(self.student.parameters(),
                                    self.teacher.parameters()):
            paddle.assign((param_k * self.m + param_q * (1. - self.m)), param_k)
            param_k.stop_gradient = True

    def train_wrapper(self, x, backbone, neck):
        # convert to list
        if not isinstance(x, (list, tuple)):
            x = [x]

        idx_crops = paddle.cumsum(paddle.unique_consecutive(
            paddle.to_tensor([inp.shape[-1] for inp in x]),
            return_counts=True,
        )[1], 0)

        start_idx, output = 0, paddle.empty((0, ))
        for end_idx in idx_crops:
            outs = backbone(paddle.concat(x[start_idx: end_idx]))
            if isinstance(outs, tuple):
                out = outs[0]
            patch_token, cls_token = out
            # accumulate outputs
            output = paddle.concat([output, cls_token])
            start_idx = end_idx

        return neck(output)


    def train_iter(self, *inputs, **kwargs):
        teacher_output = self.train_wrapper(x=inputs[:2],
                                            backbone=self.teacher[0],
                                            neck=self.teacher[1])
        student_output = self.train_wrapper(x=inputs,
                                            backbone=self.student[0],
                                            neck=self.student[1])
        
        outputs = self.head(student_output, teacher_output, self.epoch)
        return outputs


    def forward(self, *inputs, mode='train', **kwargs):
        if mode == 'train':
            return self.train_iter(*inputs, **kwargs)
        else:
            raise Exception("No such mode: {}".format(mode))
