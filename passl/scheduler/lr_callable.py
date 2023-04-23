# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

class LRCallable(object):
    pass

class CosineWithFixLR(LRCallable):
    def __init__(self,
                 learning_rate,
                 step_each_epoch,
                 epochs,
                 decay_unit='epoch',
                 **kwargs):
        self.step_each_epoch = step_each_epoch
        self.epochs = epochs
        self.lr = learning_rate
        self.decay_unit = decay_unit

    def __call__(self, group, epoch):
        """Decay the learning rate based on schedule"""
        cur_lr = self.lr * 0.5 * (1. + math.cos(math.pi * epoch / self.epochs))
        if 'fix_lr' in group and group['fix_lr']:
            group['lr'] = self.lr
        else:
            group['lr'] = cur_lr
