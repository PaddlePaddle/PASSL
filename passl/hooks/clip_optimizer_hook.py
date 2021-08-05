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
from paddle.nn import functional as F
import paddle.distributed as dist
import copy
import numpy as np

from .hook import Hook
from .builder import HOOKS
from .checkpoint_hook import CheckpointHook


@HOOKS.register()
class CLIPOptimizerHook(CheckpointHook):
    def __init__(self, 
                 interval=1,
                 by_epoch=True,
                 save_optimizer=True,
                 out_dir=None,
                 max_keep_ckpts=5,
                 priority=1,
                 **kwargs):
        super(CLIPOptimizerHook, self).__init__(interval=1,
                 by_epoch=True,
                 save_optimizer=True,
                 out_dir=None,
                 max_keep_ckpts=5,
                 priority=1,
                 **kwargs)

    def train_iter_end(self, trainer):
        for i_opt in range(len(trainer.optimizer)):
            if 'lars' in trainer.optimizer[i_opt].type:
                trainer.optimizer[i_opt].clear_gradients()
            else:
                trainer.optimizer[i_opt].clear_grad()

        outputs = trainer.outputs
        loss = outputs['loss'] 
        loss.backward()

        for i_opt in range(len(trainer.optimizer)):
            if 'lars' in trainer.optimizer[0].type:
                trainer.optimizer[i_opt].minimize(loss)
            else:
                trainer.optimizer[i_opt].step()
