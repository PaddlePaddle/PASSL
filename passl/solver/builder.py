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

from ..utils.registry import Registry, build_from_config
import paddle
import math
LRSCHEDULERS = Registry("LRSCHEDULER")
OPTIMIZERS = Registry("OPTIMIZER")


def build_lr_scheduler(cfg, iters_per_epoch):
    # FIXME: if have a better way
    if cfg.name == 'CosineAnnealingDecay':
        cfg.T_max *= iters_per_epoch
        return build_from_config(cfg, LRSCHEDULERS)
    elif cfg.name == 'MultiStepDecay':
        cfg.milestones = [x * iters_per_epoch for x in cfg.milestones]
        return build_from_config(cfg, LRSCHEDULERS)
    elif cfg.name == 'LinearWarmup':
        cfg.learning_rate = build_lr_scheduler(cfg.learning_rate, iters_per_epoch)
        cfg.warmup_steps *= iters_per_epoch
        return build_from_config(cfg, LRSCHEDULERS)
    elif cfg.name == 'CosineWarmup' or cfg.name == 'ByolLRScheduler':
        return build_from_config(cfg, LRSCHEDULERS)
    else:
        raise NotImplementedError

# To create a registry
def build_lr_scheduler_simclr(cfg, iters_per_epoch, batch_size, epochs, current_iter):
    # FIXME: if have a better way

    if cfg.name == 'CosineAnnealingDecay':
        cfg.T_max = T_max
        cfg.T_max *= iters_per_epoch
    elif cfg.name == 'MultiStepDecay':
        cfg.milestones = [x * iters_per_epoch for x in cfg.milestones]
    elif cfg.name == 'Cosinesimclr':
        cfg.iters_per_epoch = iters_per_epoch
        cfg.epochs = epochs
        cfg.T_max = T_max
    elif cfg.name == 'simclrCosineWarmup':   
        cfg.step_each_epoch = iters_per_epoch
        cfg.epochs = epochs
        cfg.warmup_steps = int(round(cfg.warmup_epochs * cfg.total_images // batch_size))
        cfg.total_steps = cfg.total_images * epochs // batch_size + 1
        cfg.T_max = cfg.total_steps - cfg.warmup_steps
        cfg.current_iter = current_iter 
        if cfg.learning_rate_scaling == 'linear':
            cfg.lr = cfg.end_lr * batch_size / 256.
        elif cfg.learning_rate_scaling == 'sqrt':
            cfg.lr = cfg.end_lr * math.sqrt(batch_size)
    return build_from_config(cfg, LRSCHEDULERS)



def build_optimizer(cfg, lr_scheduler, parameters=None):
    cfg_ = cfg.copy()
    name = cfg_.pop('name')
    if name == 'LarsMomentumOptimizer':
        return OPTIMIZERS.get(name)(lr_scheduler, parameter_list=parameters, **cfg_)
    else:
        return OPTIMIZERS.get(name)(lr_scheduler, parameters=parameters, **cfg_)


class MultiStateDictMeta(object):
    def __init__(self):
        self.metas = []

    def append(self, meta):
        self.metas.append(meta)
    
    def __getitem__(self, idx):
        return self.metas[idx]
    
    def state_dict(self):
        def convert(state_dict):
            model_dict = {}

            for k, v in state_dict.items():
                if isinstance(
                        v,
                    (paddle.fluid.framework.Variable, paddle.fluid.core.VarBase)):
                    model_dict[k] = v.numpy()
                else:
                    model_dict[k] = v

            return model_dict
        return [convert(mt.state_dict()) for mt in self.metas]
    
    def set_state_dict(self, state_dicts):
        for i, state_dict in enumerate(state_dicts):
            self.metas[i].set_state_dict(state_dict)
    
    def __len__(self):
        return len(self.metas)
