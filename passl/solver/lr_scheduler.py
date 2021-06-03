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

import math
import paddle

from paddle.optimizer.lr import MultiStepDecay
from paddle.optimizer.lr import CosineAnnealingDecay
from paddle.optimizer.lr import LinearWarmup
from .builder import LRSCHEDULERS, build_lr_scheduler


LRSCHEDULERS.register(MultiStepDecay)
LRSCHEDULERS.register(CosineAnnealingDecay)
LRSCHEDULERS.register(LinearWarmup)


class Cosine(CosineAnnealingDecay):
    """
    Cosine learning rate decay
    lr = 0.05 * (math.cos(epoch * (math.pi / epochs)) + 1)
    Args:
        lr(float): initial learning rate
        step_each_epoch(int): steps each epoch
        epochs(int): total training epochs
    """

    def __init__(self,
                 learning_rate,
                 T_max,
                 eta_min=0,
                 last_epoch=1,
                 verbose=False):
        super(Cosine, self).__init__(learning_rate,
                                     T_max,
                                     eta_min=eta_min,
                                     last_epoch=last_epoch,
                                     verbose=verbose)


    def get_lr(self):
        if self.last_epoch == 0:
            return self.base_lr
        elif (self.last_epoch - 1 - self.T_max) % (2 * self.T_max) == 0:
            return self.last_lr + (self.base_lr - self.eta_min) * (1 - math.cos(
                math.pi / self.T_max)) / 2

        return self.eta_min + 0.5 * (
            self.base_lr - self.eta_min) * (
            1 + math.cos(math.pi * self.last_epoch / self.T_max)) 


LRSCHEDULERS.register()
class CosineWarmup(LinearWarmup):
    """
    Cosine learning rate decay with warmup
    [0, warmup_epoch): linear warmup
    [warmup_epoch, epochs): cosine decay
    Args:
        lr(float): initial learning rate
        step_each_epoch(int): steps each epoch
        epochs(int): total training epochs
        warmup_epoch(int): epoch num of warmup
    """

    def __init__(self,
                 learning_rate,
                 warmup_steps,
                 start_lr,
                 end_lr,
                 T_max,
                 eta_min=0,
                 last_epoch=-1,
                 verbose=False):
        lr_sch = Cosine(learning_rate,
                        T_max,
                        eta_min=eta_min,
                        last_epoch=last_epoch,
                        verbose=verbose)

        super(CosineWarmup, self).__init__(
            learning_rate=lr_sch,
            warmup_steps=warmup_steps,
            start_lr=start_lr,
            last_epoch=last_epoch,
            end_lr=end_lr)

        self.update_specified = False

LRSCHEDULERS.register(Cosine)
LRSCHEDULERS.register(CosineWarmup)
