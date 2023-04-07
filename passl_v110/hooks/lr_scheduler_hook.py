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


@HOOKS.register()
class LRSchedulerHook(Hook):
    def __init__(self, unit='iter', priority=1):
        self.priority = priority
        assert unit in ['iter', 'epoch']
        self.unit = unit

    def train_iter_end(self, trainer):
        if self.unit == 'iter':
            trainer.lr_scheduler.step()

    def train_epoch_end(self, trainer):
        if self.unit == 'epoch':
            trainer.lr_scheduler.step()
