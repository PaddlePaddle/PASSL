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
import numpy as np

from paddle.optimizer.lr import MultiStepDecay, LRScheduler
from paddle.optimizer.lr import CosineAnnealingDecay
from paddle.optimizer.lr import LinearWarmup
from .builder import LRSCHEDULERS, build_lr_scheduler, build_lr_scheduler_simclr
from .byol_lr_scheduler import ByolLRScheduler 

LRSCHEDULERS.register(LinearWarmup)
LRSCHEDULERS.register(MultiStepDecay)
LRSCHEDULERS.register(CosineAnnealingDecay)
LRSCHEDULERS.register(ByolLRScheduler)

class Cosine(LRScheduler):
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
                 warmup_steps,
                 eta_min=0,
                 last_epoch=1,
                 verbose=False):
        super(Cosine, self).__init__(learning_rate,
                                     last_epoch=last_epoch,
                                     verbose=verbose)
        self.T_max = T_max
        self.warmup_steps = warmup_steps
        self.eta_min = eta_min
        self.last_epoch = last_epoch


    def get_lr(self):
        if self.last_epoch == 0:
            return self.base_lr
        elif (self.last_epoch - 1 - self.T_max) % (2 * self.T_max) == 0:
            return self.last_lr + (self.base_lr - self.eta_min) * (1 - math.cos(
                math.pi / self.T_max)) / 2

        return self.eta_min + 0.5 * (
            self.base_lr - self.eta_min) * (
            1 + np.cos(np.pi * self.last_epoch / (self.T_max - self.warmup_steps))) 


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
        #start_lr = 0.0
        lr_sch = Cosine(learning_rate,
                        T_max,
                        warmup_steps,
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


@LRSCHEDULERS.register()

class Cosinesimclr(LRScheduler):

    def __init__(self,
                 learning_rate,
                 T_max,
                 last_epoch=-1,
                 verbose=False):
        self.T_max = T_max
        
        super(Cosinesimclr, self).__init__(learning_rate, last_epoch,
                                                   verbose)

    def get_lr(self):
        return self.base_lr* (1 + math.cos(math.pi * self.last_epoch / self.T_max)) / 2







@LRSCHEDULERS.register()
class simclrCosineWarmup(LinearWarmup):

    def __init__(self, lr, warmup_steps, T_max, current_iter, last_epoch=-1, warmup_epoch=10, **kwargs):
        warmup_steps = warmup_steps
        T_max = T_max
        start_lr = 0.0
        lr = lr
        lr_sch = Cosinesimclr(lr, T_max, last_epoch=-1)

        super(simclrCosineWarmup, self).__init__(
            learning_rate=lr_sch,
            warmup_steps=warmup_steps,
            start_lr=start_lr,
            last_epoch=last_epoch,
            end_lr=lr)

        self.update_specified = False
        
@LRSCHEDULERS.register()      
class ViTLRScheduler(LRScheduler):    
    def __init__(self,
                 learning_rate,
                 T_max,
                 decay_type='cosine',
                 linear_end=1e-5,
                 warmup_steps=0,
                 verbose=False,
                 last_epoch=-1,
                 **kwargs):

        self.base_lr = learning_rate
        self.linear_end = linear_end
        self.T_max = T_max
        self.warmup_steps = warmup_steps
        
        if self.warmup_steps >= self.T_max:
            self.warmup_steps = self.T_max
        
        self.decay_type = decay_type
        self.last_epoch = last_epoch
        super(ViTLRScheduler, self).__init__(learning_rate, last_epoch, verbose)   

    def get_lr(self):
        
        progress = (self.last_epoch - self.warmup_steps) / float(self.T_max - self.warmup_steps)
        progress = min(1.0, max(0.0, progress))
        
        if self.decay_type == 'linear':
            lr = self.linear_end + (self.base_lr - self.linear_end) * (1.0 - progress)
        elif self.decay_type == 'cosine':
            lr = 0.5 * self.base_lr * (1.0 + math.cos(math.pi * progress))
        if self.warmup_steps:
            lr = lr * min(1.0, self.last_epoch / self.warmup_steps)
            
        return lr
    
@LRSCHEDULERS.register()
class TimmCosine(LRScheduler):    
    def __init__(self,
                 learning_rate,
                 T_max,
                 eta_min=0.0,
                 warmup_epoch=0,
                 warmup_start_lr=0.0,
                 verbose=False,
                 last_epoch=-1,
                 **kwargs):
        if warmup_epoch >= T_max:
            warmup_epoch = T_max
            
        self.learning_rate = learning_rate

        self.T_max = T_max
        self.warmup_steps = warmup_epoch
        
        self.eta_min = eta_min
        self.last_epoch = last_epoch
        self.warmup_start_lr = warmup_start_lr
        
        super(TimmCosine, self).__init__(learning_rate, last_epoch, verbose)   

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            return float(self.last_epoch) * (self.learning_rate - self.warmup_start_lr) / float(self.warmup_steps) + self.warmup_start_lr
        
        cur_steps = self.last_epoch - (self.T_max * (self.last_epoch // self.T_max))
        return self.eta_min + 0.5 * (self.base_lr - self.eta_min) * (1 + math.cos(math.pi * cur_steps / self.T_max))































LRSCHEDULERS.register(Cosine)
LRSCHEDULERS.register(CosineWarmup)

