# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import time
import collections
import platform
import paddle
from passl.core import grad_sync, param_sync
from passl.utils import io

from passl.utils import profiler
from passl.utils import logger
from .loop import TrainingEpochLoop

class ContrastiveLearningTrainingEpochLoop(TrainingEpochLoop):

    def __init__(self, trainer, epochs, max_train_step=None, val_loop=None):
        super().__init__(trainer, epochs, max_train_step=max_train_step, val_loop=val_loop)

    def forward_backward(self, batch):
        # Gradient Merge(GuoxiaWang): Accumulate gradient over multiple
        # steps to save on memory.

        self.batch_size = batch[0].shape[0]
        assert self.batch_size % self.trainer.accum_steps == 0, f'Bad accum_steps {self.trainer.accum_steps} for batch size {self.batch_size}. This may be caused by two reasons: 1) the batch size setting is unreasonable and cannot be divisible, 2) drop_last in the sampler configuration is not set to True.'
        step_size = self.batch_size // self.trainer.accum_steps

        final_loss_dict = collections.defaultdict(float)

        for idx in range(self.trainer.accum_steps):
            sub_batch = [b[idx * step_size:(idx + 1) * step_size] for b in batch]

            # do cast if using fp16 otherwise do nothing
            with paddle.amp.auto_cast(
                    enable=self.trainer.fp16,
                    custom_white_list=self.trainer.fp16_custom_white_list,
                    custom_black_list=self.trainer.fp16_custom_black_list,
                    level=self.trainer.fp16_level):

                loss_dict = self.trainer.model(sub_batch)
                if isinstance(loss_dict, paddle.Tensor):
                    loss_dict = {'loss': loss_dict}

            for key in loss_dict:
                loss_dict[key] = loss_dict[key] / self.trainer.accum_steps

                with paddle.no_grad():
                    final_loss_dict[key] += loss_dict[key]

            # loss scaling if using fp16 otherwise do nothing
            scaled = self.trainer.scaler.scale(loss_dict["loss"])
            scaled.backward()

        return final_loss_dict

    def train_one_step(self, batch):

        # remove label
        batch = batch[0]

        # do forward and backward
        loss_dict = self.forward_backward(batch)

        grad_sync(self.trainer.optimizer.param_groups)

        # do unscale and step if using fp16 and not found nan/inf
        # otherwise do nothing
        self.trainer.scaler.step(self.trainer.optimizer)
        # do update loss scaling if using fp16
        # otherwise do nothing
        self.trainer.scaler.update()
        # clear gradients
        self.trainer.optimizer.clear_grad()

        if self.trainer.lr_decay_unit == 'step':
            self.trainer.optimizer.lr_step(self.global_step)

        return None, loss_dict
