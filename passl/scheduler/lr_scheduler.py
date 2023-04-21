# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
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
from __future__ import unicode_literals

import math
from paddle.optimizer import lr
from passl.utils import logger


class TimmCosine(lr.LRScheduler):
    def __init__(self,
                 learning_rate,
                 step_each_epoch,
                 epochs,
                 decay_unit='epoch',
                 eta_min=0.0,
                 warmup_epoch=0,
                 warmup_start_lr=0.0,
                 warmup_prefix=False,
                 verbose=False,
                 last_epoch=-1,
                 **kwargs):
        if warmup_epoch >= epochs:
            msg = f"When using warm up, the value of \"Global.epochs\" must be greater than value of \"Optimizer.lr.warmup_epoch\". The value of \"Optimizer.lr.warmup_epoch\" has been set to {epochs}."
            logger.warning(msg)
            warmup_epoch = epochs
        self.learning_rate = learning_rate
        assert decay_unit in ['step', 'epoch']
        self.decay_unit = decay_unit
        if decay_unit == 'step':
            self.T_max = epochs * step_each_epoch
            self.warmup_steps = int(round(warmup_epoch * step_each_epoch))
        else:
            self.T_max = epochs
            self.warmup_steps = warmup_epoch

        self.eta_min = eta_min
        self.last_epoch = last_epoch
        self.warmup_start_lr = warmup_start_lr
        self.warmup_prefix = warmup_prefix

        if not isinstance(learning_rate, (float, int)):
            raise TypeError(
                "The type of learning rate must be float, but received {}".
                format(type(learning_rate)))
        self.base_lr = float(learning_rate)
        self.last_lr = float(learning_rate)
        self.last_epoch = last_epoch
        self.verbose = verbose
        self._var_name = None

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            return float(max(0, self.last_epoch)) * (
                self.learning_rate - self.warmup_start_lr
            ) / float(self.warmup_steps) + self.warmup_start_lr

        last_epoch = self.last_epoch
        T_max = self.T_max
        if self.warmup_prefix:
            last_epoch = last_epoch - self.warmup_steps
            T_max = self.T_max - self.warmup_steps
        cur_steps = last_epoch - (self.T_max * (last_epoch // self.T_max))
        return self.eta_min + 0.5 * (self.base_lr - self.eta_min) * (
            1 + math.cos(math.pi * cur_steps / T_max))


class ViTLRScheduler(lr.LRScheduler):
    def __init__(self,
                 learning_rate,
                 step_each_epoch,
                 epochs,
                 decay_type='cosine',
                 linear_end=1e-5,
                 warmup_steps=0,
                 verbose=False,
                 last_epoch=-1,
                 **kwargs):

        self.linear_end = linear_end
        self.T_max = epochs * step_each_epoch
        self.warmup_steps = warmup_steps

        if self.warmup_steps >= self.T_max:
            self.warmup_steps = self.T_max

        self.decay_type = decay_type
        self.last_epoch = last_epoch
        super(ViTLRScheduler, self).__init__(learning_rate, last_epoch,
                                             verbose)

    def get_lr(self):

        progress = (self.last_epoch - self.warmup_steps
                    ) / float(self.T_max - self.warmup_steps)
        progress = min(1.0, max(0.0, progress))

        if self.decay_type == 'linear':
            lr = self.linear_end + (self.base_lr - self.linear_end) * (
                1.0 - progress)
        elif self.decay_type == 'cosine':
            lr = 0.5 * self.base_lr * (1.0 + math.cos(math.pi * progress))
        if self.warmup_steps:
            lr = lr * min(1.0, self.last_epoch / self.warmup_steps)

        return lr


class Step(lr.LRScheduler):
    def __init__(self,
                 step_each_epoch,
                 epochs,
                 boundaries, # [12, 16]
                 values,    #[0.01, 0.002, 0.0004],
                 warmup_steps=0,
                 warmup_epochs=0,
                 decay_unit='epoch',
                 warmup_start_lr=0.0,
                 warmup_end_lr=0.0,
                 last_epoch=-1,
                 verbose=False,
                 **kwargs):

        assert decay_unit in ['step', 'epoch']
        self.decay_unit = decay_unit
        if decay_unit == 'step':
            self.T_max = epochs * step_each_epoch
            self.warmups = warmup_steps
        else:
            self.T_max = epochs
            self.warmups = warmup_epochs

        self.warmup_start_lr = warmup_start_lr
        self.warmup_end_lr = warmup_end_lr

        self.boundaries = boundaries
        self.values = values
        super(Step, self).__init__(last_epoch=last_epoch, verbose=verbose)

    def get_lr(self):
        if self.last_epoch < self.warmups:
            return (self.warmup_end_lr - self.warmup_start_lr) * float(
                self.last_epoch) / float(self.warmups) + self.warmup_start_lr

        for i in range(len(self.boundaries)):
            if self.last_epoch < self.boundaries[i]:
                return self.values[i]
        return self.values[len(self.values) - 1]


class Poly(lr.LRScheduler):
    def __init__(self,
                 step_each_epoch,
                 epochs,
                 learning_rate,
                 warmup_steps=0,
                 warmup_epochs=0,
                 decay_unit='epoch',
                 warmup_start_lr=0.0,
                 warmup_end_lr=0.0,
                 last_epoch=-1,
                 verbose=False,
                 **kwargs):

        assert decay_unit in ['step', 'epoch']
        self.decay_unit = decay_unit
        if decay_unit == 'step':
            self.T_max = epochs * step_each_epoch
            if warmup_steps == 0 and warmup_epochs > 0:
                self.warmups = warmup_epochs * step_each_epoch
            else:
                self.warmups = warmup_steps
        else:
            self.T_max = epochs
            self.warmups = warmup_epochs

        self.warmup_start_lr = warmup_start_lr
        self.warmup_end_lr = warmup_end_lr

        super(Poly, self).__init__(
            learning_rate, last_epoch=last_epoch, verbose=verbose)

    def get_lr(self):
        if self.last_epoch < self.warmups:
            return (self.warmup_end_lr - self.warmup_start_lr) * float(
                self.last_epoch) / float(self.warmups) + self.warmup_start_lr

        return self.base_lr * pow(1 - float(self.last_epoch - self.warmups) /
                                  float(self.T_max - self.warmups), 2)
