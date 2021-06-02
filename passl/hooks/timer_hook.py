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

import time

from .hook import Hook
from .builder import HOOKS
from ..utils import AverageMeter


@HOOKS.register()
class IterTimerHook(Hook):
    def __init__(self, priority=1):
        self.priority = priority
        
    def epoch_begin(self, runner):
        self.t = time.time()

    def iter_begin(self, runner):
        if 'data_time' not in runner.logs:
            runner.logs['data_time'] = AverageMeter('data_time')
        runner.logs['data_time'].update(time.time() - self.t)

    def iter_end(self, runner):
        if 'time' not in runner.logs:
            runner.logs['time'] = AverageMeter('time')
        runner.logs['time'].update(time.time() - self.t)
        self.t = time.time()
