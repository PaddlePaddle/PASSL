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
import copy
import re
from collections import defaultdict
from passl.utils import logger


def group_with_matcher(model, group_matcher):
    """

    Args:
        named_params: List like [(name, param),]
        group_matcher: Dict like {group_name: regular_expression1}
    Returns:
        param_groups: Dict like {group_name: [param_name1, param_name2, ...]}

    """
    matcher_list = []
    for group_name, re_exps in group_matcher.items():
        assert re_exps is not None, "re_exps should not be None."
        if isinstance(re_exps, (tuple, list)):
            for re_str in re_exps:
                matcher_list.append((group_name, re.compile(re_str)))
        else:
            matcher_list.append((group_name, re.compile(re_exps)))
    param_groups = defaultdict(list)
    default_group = []
    for name, param in model.named_parameters():
        if param.stop_gradient:
           continue
        flag = 0
        for group_name, matcher in matcher_list:
            res = matcher.match(name)
            if res:
                param_groups[group_name].append((name, param))
                flag = 1
        if flag == 0:
            default_group.append((name, param))
    if len(default_group) > 0:
        param_groups['default'] = default_group
    param_groups = {k: {"params": v} for k, v in param_groups.items()}
    return param_groups


def group_params_by_state(param_groups_map):
    '''
    group parameters by state for tensor fusion
    Args:
        param_groups_map: Dict like {'group_name': {'params': [param1, param2, ...]}}

    Returns:
        new_param_groups: Dict like {'group_name': {'params': [param1, param2, ...]}}
    '''
    new_param_groups = {}
    for g_name in param_groups_map:
        for param in param_groups_map[g_name]['params']:
            if param.stop_gradient:
                continue
            state = copy.deepcopy(param.__dict__)
            new_group_name = g_name+'_'+str(state)
            if new_group_name not in new_param_groups:
                new_param_groups[new_group_name] = {
                    "params": [],
                    "group_name": new_group_name,
                }
                for key in param_groups_map[g_name]:
                    if key not in ["params", "group_name"]:
                        new_param_groups[new_group_name][key] = param_groups_map[g_name][key]

            new_param_groups[new_group_name]["params"].append(param)
    logger.info(f"The original param_groups which has {len(param_groups_map)} "
                f"groups has been split to {len(new_param_groups)} groups by state.")
    return new_param_groups


def param_group_layer_decay(
        model,
        layer_decay,
        weight_decay=None,
        group_matcher=None,
        no_weight_decay_list=(),
        param_groups_map=None,
    ):
    '''
    group parameters by layer_decay and weight_decay setting
    Args:
        model: instance of paddle.nn.Layer
        layer_decay: float or None
        weight_decay: float or None by default, which can also assigned in the optimizer args,
                    but it has the highest priority if given here.
        group_matcher: Dict like {group_name: regular_expression1}
        no_weight_decay_list: list of string(layer name keyword)
        param_groups_map:  Dict like {group_name: {'params': [(name, group), ...]}}

    Returns:
        param_groups: Dict like {group_name: {'params': [(name, group), ...]}}
    '''
    assert (not group_matcher) or (not param_groups_map), \
        "group_matcher and param_names_group should not be given in the same time."
    if group_matcher:
        param_groups_map = group_with_matcher(model, group_matcher)
    num_layers = len(param_groups_map)
    layer_scales = {z[0]: layer_decay ** (num_layers - i) for i, (k, v) in enumerate(param_groups_map.items()) for z in v}
    param_groups = {}
    for g_name in param_groups_map:
        for name, param in param_groups_map[g_name]['params']:
            if param.stop_gradient:
                continue
            lr_scale = layer_scales[name] if name in layer_scales else 1.
            if any(nd in name for nd in no_weight_decay_list):
                this_decay = 0.
                g_decay = "no_weight_decay"
            else:
                this_decay = weight_decay
                g_decay = "weight_decay"
            new_group_name = g_name + '_' + g_decay
            if new_group_name not in param_groups:
                param_groups[new_group_name] = {
                    "lr_scale": lr_scale,
                    "params": [],
                    "group_name": new_group_name,
                }
                for key in param_groups_map[g_name]:
                    if key not in param_groups[new_group_name]:
                        param_groups[new_group_name][key] = param_groups_map[g_name][key]
            if this_decay is not None:
                param_groups[new_group_name]["weight_decay"] = this_decay
            param_groups[new_group_name]["params"].append((name, param))
    return param_groups


def param_group_weight_decay(
        model,
        group_matcher=None,
        weight_decay=None,
        no_weight_decay_list=(),
        param_groups_map=None,
    ):
    '''
    group parameters by weight_decay setting
    Args:
        model: instance of paddle.nn.Layer
        group_matcher: Dict like {group_name: regular_expression1}
        weight_decay: float or None by default, which can also assigned in the optimizer args,
                    but it has the highest priority if given here.
        no_weight_decay_list: list of string(layer name keyword)
        param_groups_map: Dict like {group_name: {'params': [(name, group), ...]}}

    Returns:
        param_groups: Dict like {group_name: {'params': [(name, group), ...]}}
    '''
    # weight_decay value can be None and assigned in the optimizer config,
    # but it has the highest priority if given here.
    assert (not group_matcher) or (not param_groups_map), \
        "group_matcher and param_names_group should not be given in the same time."
    param_groups = {}
    if group_matcher is not None:
        param_groups_map = group_with_matcher(model, group_matcher)
    for g_name in param_groups_map:
        for name, param in param_groups_map[g_name]['params']:
            if param.stop_gradient:
                continue
            if any(nd in name for nd in no_weight_decay_list):
                g_decay = "no_weight_decay"
                this_decay = 0.
            else:
                g_decay = "weight_decay"
                this_decay = weight_decay
            new_group_name = g_name + "_" + g_decay
            if new_group_name not in param_groups:
                param_groups[new_group_name] = {
                    "params": [],
                    "group_name": new_group_name,
                }
                for key in param_groups_map[g_name]:
                    if key not in param_groups[new_group_name]:
                        param_groups[new_group_name][key] = param_groups_map[g_name][key]
            if this_decay is not None:
                param_groups[new_group_name]["weight_decay"] = this_decay
            param_groups[new_group_name]["params"].append((name, param))

    return param_groups
