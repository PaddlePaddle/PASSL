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

from .hook import Hook
from .builder import HOOKS
from ..solver.lr_scheduler import Cosine
from paddle.fluid.regularizer import L2Decay

@HOOKS.register()
class DINOHook(Hook):
    def __init__(self,
                 priority=1,
                 weight_decay=0.04,
                 weight_decay_end=0.4,
                 momentum=0.996,
                 momentum_end=1.0,
                 total_steps=250200,
                 ):
        self.priority = priority
        self.wd_schedule = Cosine(
            weight_decay, total_steps, warmup_steps=0,
            eta_min=weight_decay_end, last_epoch=-1)
        self.mm_schedule = Cosine(
            momentum, total_steps, warmup_steps=0,
            eta_min=momentum_end, last_epoch=-1)

    def train_iter_begin(self, trainer):
        # update weight decay
        self.wd_schedule.step()
        cur_wd = self.wd_schedule.get_lr()
        trainer.optimizer.regularization = L2Decay(cur_wd)

        # update teacher momentum
        self.mm_schedule.step()
        cur_m = self.mm_schedule.get_lr()

        # update epoch
        cur_epoch = trainer.current_epoch

        if hasattr(trainer.model, '_layers'):
            trainer.model._layers.m = cur_m
            trainer.model._layers.epoch = cur_epoch
        else:
            trainer.model.m = cur_m
            trainer.model.epoch = cur_epoch
