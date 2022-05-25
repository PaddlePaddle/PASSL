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

import numpy as np

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import paddle.distributed as dist

from .builder import HEADS


@HEADS.register()
class DINOHead(nn.Layer):
    def __init__(self, out_dim, n_crops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, n_epochs, student_temp=0.1,
                 center_momentum=0.9):
        super().__init__()
        self.n_crops = n_crops
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.register_buffer('center', paddle.ones((1, out_dim)))
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(n_epochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, student_output, teacher_output, epoch):
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.n_crops)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, axis=-1)
        teacher_out = teacher_out.detach().chunk(2)

        outputs = dict()
        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                loss = paddle.sum(-q * F.log_softmax(student_out[v], axis=-1), axis=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        outputs['loss'] = total_loss
        self.update_center(teacher_output)
        return outputs

    @paddle.no_grad()
    def update_center(self, teacher_output):
        batch_center = paddle.sum(teacher_output, axis=0, keepdim=True)
        if dist.get_world_size() >= 2:
            dist.all_reduce(batch_center)
        batch_center = batch_center / (len(teacher_output) * dist.get_world_size())

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)

