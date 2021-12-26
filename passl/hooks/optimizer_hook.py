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
class OptimizerHook(Hook):
    def __init__(self, priority=1):
        self.priority = priority
        
    def train_iter_end(self, trainer):
        if 'Lars' in trainer.cfg['optimizer']['name']:
            trainer.optimizer.clear_gradients()
        else:
            trainer.optimizer.clear_grad()

        loss = 0
        loss = trainer.outputs['loss']
        
        if trainer.use_amp:
            scaled_loss = trainer.scaler.scale(loss)
            scaled_loss.backward()
            trainer.scaler.step(trainer.optimizer)
            trainer.scaler.update()

        else:
            loss.backward()
            if 'lars' in trainer.optimizer.type:
                trainer.optimizer.minimize(loss)
            else:
                trainer.optimizer.step()

        if 'loss' not in trainer.outputs:
            trainer.outputs['loss'] = loss
