# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import defaultdict

import copy
import re
import paddle

from passl.core.grad_clip import ClipGradByGlobalNorm
from passl.core.param_fuse import get_fused_params
from passl.scheduler import build_lr_scheduler, LRCallable
from passl.utils import logger

from .optimizer import Optimizer
from .adamw import AdamW
from .adafactor import Adafactor
from .momentum import Momentum
from .momentum_lars import MomentumLARS
from .momentum_larc import MomentumLARC
from .adan import Adan
from .utils.group_params import (
    param_group_layer_decay,
    param_group_weight_decay,
    group_params_by_state)


def build_group_lr_scheduler(param_groups_cfg, epochs, step_each_epoch, lr_decay_unit):
    '''
    Build lr scheduler in each param_group.
    Args:
        param_groups_cfg: Dict, param_groups config
        epochs: Int, epochs
        step_each_epoch: Int, step for each epoch

    Returns:
        param_groups_cfg: Dict of param_groups config in which lr has beed build
    '''
    for idx, item in enumerate(param_groups_cfg):
        lr_cfg = item.get('lr', None)
        if isinstance(lr_cfg, dict):
            if 'decay_unit' in lr_cfg:
                logger.warning('decay_unit is no need to set, for it will be reset by lr_decay_unit.')
            lr_cfg['decay_unit'] = lr_decay_unit
            lr_scheduler = build_lr_scheduler(lr_cfg, epochs, step_each_epoch)
            if isinstance(lr_scheduler, LRCallable):
                item['lr_func'] = lr_scheduler
            else:
                item['lr'] = lr_scheduler
        elif isinstance(lr_cfg, float):
           item['lr'] = lr_cfg
        logger.info('build lr scheduler in param_groups succeed.')
    return param_groups_cfg


def group_params(model, param_groups_cfg=None):
    '''
    Group params by config or by stop_gradient by default.
    Args:
        model: paddle.nn.Layer
        param_groups_cfg: Dict, param_groups config
    Returns:
        Dict, f.g. {'group_name': {'params': [(name, param), ...],}}
    '''

    if param_groups_cfg and len(param_groups_cfg) > 0:
        params_dict = {}
        # init params_dict by config
        for group in param_groups_cfg:
            params_dict[group['name']] = {}
            params_dict[group['name']]['params'] = []
            for k, v in group.items():
                params_dict[group['name']][k] = v
        # add params
        for name, param in model.named_parameters():
            if param.stop_gradient:
                continue
            flag = 0
            for g_name in params_dict:
                if 'regular_exp' in params_dict[g_name]:
                    regular_exp = params_dict[g_name]['regular_exp']
                    group_matcher = re.compile(regular_exp)
                else:
                    group_matcher = re.compile(g_name)
                if group_matcher.match(name):
                    params_dict[g_name]["params"].append((name, param))
                    flag = 1
                    break
            if flag == 0:
                if 'default' not in params_dict:
                    params_dict['default'] = {'params': []}
                params_dict['default']["params"].append((name, param))

        logger.info(f'Model parameters has been split into {len(params_dict)} groups by config.')
        for key in params_dict:
            logger.info(f"{key}-params length: {len(params_dict[key]['params'])}")

        return params_dict

    # default group method
    param_groups = []
    for name, param in model.named_parameters():
        if param.stop_gradient:
            continue
        param_groups.append((name, param))
    logger.info(f'Model parameters has been split into 1 groups by default.')
    return {'default': {"params": param_groups}}


def build_optimizer(config, lr_scheduler, model, epochs, step_each_epoch, lr_decay_unit):
    config = copy.deepcopy(config)

    optim_name = config.pop('name')
    layer_decay = config.pop('layer_decay', None)
    grad_clip = None
    grad_clip_config = config.pop('grad_clip', None)
    if grad_clip_config is not None:
        grad_clip_name = grad_clip_config.pop('name', 'ClipGradByGlobalNorm')
        grad_clip = eval(grad_clip_name)(**grad_clip_config)

    weight_decay = config.get('weight_decay', None)
    no_weight_decay_name = config.pop('no_weight_decay_name', [])

    tensor_fusion = config.pop('tensor_fusion', True)
    if 'LAR' in optim_name:
        tensor_fusion = False
        logger.info('LARS or LARC Optimizer can not use tensor fusion technology. '
                    'It automatically fall back to `tensor_fusion = False`.')

    # param_groups is a dict like {'group_name': {'params': [(name, param), ...]}}
    if hasattr(model, 'param_group_fn'):
        # param groups are defined by model
        model_group_cfg = config.pop('param_group_fn', {})
        param_group_map = model.param_group_fn(no_weight_decay_name=no_weight_decay_name, weight_decay=weight_decay,
                                               layer_decay=layer_decay, **model_group_cfg)
    else:
        param_groups_cfg = config.get('param_groups', None)
        if param_groups_cfg and len(param_groups_cfg) > 0:
            param_groups_cfg = build_group_lr_scheduler(param_groups_cfg, epochs, step_each_epoch, lr_decay_unit)
        param_group_map = group_params(model, param_groups_cfg)
        if isinstance(layer_decay, float):
            param_group_map = param_group_layer_decay(model,
                                                      layer_decay,
                                                      weight_decay=weight_decay,
                                                      param_groups_map=param_group_map,
                                                      no_weight_decay_list=no_weight_decay_name,
                                                      )
        elif len(no_weight_decay_name) > 0:
            param_group_map = param_group_weight_decay(model,
                                                      weight_decay=weight_decay,
                                                      param_groups_map=param_group_map,
                                                      no_weight_decay_list=no_weight_decay_name,
                                                      )

    for key in param_group_map:
        param_group_map[key]['params'] = [p for (n, p) in param_group_map[key]['params']]

    if tensor_fusion:
        param_group_map = group_params_by_state(param_group_map)
        # fuse params
        for key in param_group_map:
            if 'gpu' not in paddle.get_device():
                continue
            if "'is_distributed': True" in key:
                continue
            if "'has_sparse_grad': True" in key:
                continue
            param_group_map[key]["params"] = get_fused_params(param_group_map[key]["params"])

    param_group = []
    for key in param_group_map:
        group = param_group_map[key]
        if "'is_distributed': True" in key:
            group['is_distributed'] = True
        if 'no_weight_decay' in key:
            group['weight_decay'] = 0.0
        param_group.append(group)

    # build default lr scheduler
    lr = lr_scheduler
    lr_func = None
    lr_cfg = config.pop('lr', None)
    if isinstance(lr_cfg, float):
        lr = lr_cfg
    elif isinstance(lr_cfg, dict):
        if 'decay_unit' in lr_cfg:
            logger.warning('decay_unit is no need to set, for it will be reset by lr_decay_unit.')
        lr_cfg['decay_unit'] = lr_decay_unit
        lr_scheduler = build_lr_scheduler(lr_cfg, epochs, step_each_epoch)
        lr = lr_scheduler
    if isinstance(lr_scheduler, LRCallable):
        lr = lr_scheduler.lr
        lr_func = lr_scheduler
    assert lr is not None, 'lr should not be None.'
    optim = eval(optim_name)(param_group,
                             lr=lr,
                             lr_func=lr_func,
                             grad_clip=grad_clip,
                             **config)
    logger.debug("build optimizer ({}) success..".format(optim))
    return optim
