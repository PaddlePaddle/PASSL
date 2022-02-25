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
import copy
import paddle
from paddle.nn.clip import ClipGradByGlobalNorm, ClipGradByNorm

from ..utils.registry import Registry, build_from_config

LRSCHEDULERS = Registry("LRSCHEDULER")
OPTIMIZERS = Registry("OPTIMIZER")


def build_lr_scheduler(cfg, iters_per_epoch):
    # FIXME: if have a better way
    if cfg.name == 'CosineAnnealingDecay' or cfg.name == 'ViTLRScheduler':
        cfg.T_max *= iters_per_epoch
        return build_from_config(cfg, LRSCHEDULERS)
    elif cfg.name == 'MultiStepDecay':
        cfg.milestones = [x * iters_per_epoch for x in cfg.milestones]
        return build_from_config(cfg, LRSCHEDULERS)
    elif cfg.name == 'LinearWarmup':
        cfg.learning_rate = build_lr_scheduler(cfg.learning_rate,
                                               iters_per_epoch)
        cfg.warmup_steps *= iters_per_epoch
        return build_from_config(cfg, LRSCHEDULERS)
    elif cfg.name == 'CosineWarmup' or cfg.name == 'ByolLRScheduler' or cfg.name == 'TimmCosine':
        return build_from_config(cfg, LRSCHEDULERS)
    else:
        raise NotImplementedError


# To create a registry
def build_lr_scheduler_simclr(cfg, iters_per_epoch, batch_size, epochs,
                              current_iter):
    # FIXME: if have a better way

    if cfg.name == 'CosineAnnealingDecay':
        cfg.T_max *= iters_per_epoch
    elif cfg.name == 'MultiStepDecay':
        cfg.milestones = [x * iters_per_epoch for x in cfg.milestones]
    elif cfg.name == 'Cosinesimclr':
        cfg.iters_per_epoch = iters_per_epoch
        cfg.epochs = epochs
    elif cfg.name == 'simclrCosineWarmup':
        cfg.step_each_epoch = iters_per_epoch
        cfg.epochs = epochs
        cfg.warmup_steps = int(
            round(cfg.warmup_epochs * cfg.total_images // batch_size))
        cfg.total_steps = cfg.total_images * epochs // batch_size + 1
        cfg.T_max = cfg.total_steps - cfg.warmup_steps
        cfg.current_iter = current_iter
        if cfg.learning_rate_scaling == 'linear':
            cfg.lr = cfg.end_lr * batch_size / 256.
        elif cfg.learning_rate_scaling == 'sqrt':
            cfg.lr = cfg.end_lr * math.sqrt(batch_size)
    return build_from_config(cfg, LRSCHEDULERS)


def build_clip_optimizer(cfg, lr_scheduler, parameters=None):
    cfg = copy.deepcopy(cfg)
    name = cfg.pop('name')

    # step1 clip grad
    if 'grad_clip' in cfg:
        grad_clip_cfg = cfg.pop('grad_clip')
        if grad_clip_cfg['name'] == 'global_norm':
            clip_norm = grad_clip_cfg['value']
            cfg['grad_clip'] = ClipGradByGlobalNorm(clip_norm=clip_norm)
        elif grad_clip_cfg['name'] == 'clip_norm':
            clip_norm = grad_clip_cfg['value']
            cfg['grad_clip'] = ClipGradByNorm(clip_norm=clip_norm)

    # step2 Adapt Lars and Lamb optimizer parameter argument.
    if 'Lars' in name or 'Lamb' in name:
        cfg['parameter_list'] = parameters
    else:
        cfg['parameters'] = parameters
    return OPTIMIZERS.get(name)(lr_scheduler, **cfg)


def get_num_layer_for_vit(var_name, num_max_layer):
    if var_name in ("backbone.cls_token", "backbone.mask_token",
                    "backbone.pos_embed"):
        return 0
    elif var_name.startswith("backbone.patch_embed"):
        return 0
    elif var_name.startswith("backbone.rel_pos_bias"):
        return num_max_layer - 1
    elif var_name.startswith("backbone.blocks"):
        layer_id = int(var_name.split('.')[2])
        return layer_id + 1
    else:
        return num_max_layer - 1


class LayerDecayValueAssigner(object):
    def __init__(self, values):
        self.values = values

    def get_scale(self, layer_id):
        return self.values[layer_id]

    def get_layer_id(self, var_name):
        return get_num_layer_for_vit(var_name, len(self.values))


def get_parameter_groups(cfg,
                         model,
                         skip_list=(),
                         get_num_layer=None,
                         get_layer_scale=None):
    weight_decay = cfg['weight_decay']
    parameter_group_names = {}
    parameter_group_vars = {}

    for name, param in model.named_parameters():
        if param.stop_gradient:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            group_name = "no_decay"
            this_weight_decay = 0.
        else:
            group_name = "decay"
            this_weight_decay = weight_decay
        if get_num_layer is not None:
            layer_id = get_num_layer(name)
            group_name = "layer_%d_%s" % (layer_id, group_name)
        else:
            layer_id = None

        if group_name not in parameter_group_names:
            if get_layer_scale is not None:
                scale = get_layer_scale(layer_id)
            else:
                scale = 1.

            parameter_group_names[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
                "learning_rate": scale
            }
            parameter_group_vars[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
                "learning_rate": scale
            }
        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(name)
    return list(parameter_group_vars.values())


def build_optimizer(cfg, lr_scheduler, model_list=None):
    cfg = copy.deepcopy(cfg)
    name = cfg.pop('name')
    if 'layer_decay' in cfg:
        layer_decay = cfg.pop('layer_decay')
        assert isinstance(layer_decay, float)
    if layer_decay is None:
        layer_decay = 1.0

    # step 1 clip grad
    if 'grad_clip' in cfg:
        grad_clip_cfg = cfg.pop('grad_clip')
        if grad_clip_cfg['name'] == 'global_norm':
            clip_norm = grad_clip_cfg['value']
            cfg['grad_clip'] = ClipGradByGlobalNorm(clip_norm=clip_norm)
        elif grad_clip_cfg['name'] == 'clip_norm':
            clip_norm = grad_clip_cfg['value']
            cfg['grad_clip'] = ClipGradByNorm(clip_norm=clip_norm)

    if layer_decay < 1.0:
        num_layers = model_list[0].backbone.get_num_layers()
        assigner = LayerDecayValueAssigner(
            list(layer_decay**(num_layers + 1 - i)
                 for i in range(num_layers + 2)))
    else:
        assigner = None
    if assigner is not None:
        parameters = get_parameter_groups(cfg,
                                          model_list[0],
                                          get_num_layer=assigner.get_layer_id,
                                          get_layer_scale=assigner.get_scale)
    else:
        parameters = sum([m.parameters()
                          for m in model_list], []) if model_list else None

    # step 2 Adapt Lars and Lamb optimizer parameter argument.
    if 'Lars' in name or 'Lamb' in name:
        cfg['parameter_list'] = parameters
    else:
        cfg['parameters'] = parameters

        # exclude weight decay
        def _apply_decay_param_fun(name):
            return name not in exclude_from_weight_decay_list

        if 'exclude_from_weight_decay' in cfg:
            ex_decay_cfg = cfg.pop('exclude_from_weight_decay')
            exclude_from_weight_decay_list = [
                p.name for model in model_list
                for n, p in model.named_parameters()
                if any(nd in n for nd in ex_decay_cfg)
            ]
            cfg['apply_decay_param_fun'] = _apply_decay_param_fun

    return OPTIMIZERS.get(name)(lr_scheduler, **cfg)
