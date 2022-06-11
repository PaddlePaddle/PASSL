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
from ..solver.builder import build_optimizer


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
            if 'lars' in trainer.optimizer.type:
                trainer.scaler.minimize(trainer.optimizer, scaled_loss)
            else:
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


@HOOKS.register()
class DINOOptimizerHook(Hook):
    def __init__(self, freeze_last_layer, priority=1):
        self.priority = priority
        self.freeze_last_layer = freeze_last_layer

    def run_begin(self, trainer):
        if hasattr(trainer.model, '_layers'):
            model = trainer.model._layers
        else:
            model = trainer.model

        # build dino optimizer
        trainer.optimizer = build_optimizer(
            trainer.cfg.optimizer, trainer.lr_scheduler,
            [model.student[0], model.student[1].mlp])
        trainer.last_layer_optimizer = build_optimizer(
            trainer.cfg.optimizer, trainer.lr_scheduler,
            [model.student[1].last_layer])
        
    def train_iter_end(self, trainer):
        if 'Lars' in trainer.cfg['optimizer']['name']:
            trainer.optimizer.clear_gradients()
            trainer.last_layer_optimizer.clear_gradients()
        else:
            trainer.optimizer.clear_grad()
            trainer.last_layer_optimizer.clear_grad()

        loss = 0
        loss = trainer.outputs['loss']

        # backward
        if trainer.use_amp:
            scaled_loss = trainer.scaler.scale(loss)
            scaled_loss.backward()
        else:
            loss.backward()

        # cancel gradients for the prototypes
        if hasattr(trainer.model, '_layers'):
            model = trainer.model._layers
        else:
            model = trainer.model

        if trainer.current_epoch < self.freeze_last_layer:
            trainer.last_layer_optimizer.clear_grad()

        # update student parameters
        if trainer.use_amp:
            if 'lars' in trainer.optimizer.type:
                trainer.scaler.minimize(trainer.optimizer, scaled_loss)
            else:
                trainer.scaler.step(trainer.optimizer)
                trainer.scaler.update()
        else:
            if 'lars' in trainer.optimizer.type:
                trainer.optimizer.minimize(loss)
            else:
                trainer.optimizer.step()

            if trainer.current_epoch >= self.freeze_last_layer:
                trainer.last_layer_optimizer.step()

        # EMA update for the teacher
        model.momentum_update_teacher()

        if 'loss' not in trainer.outputs:
            trainer.outputs['loss'] = loss
