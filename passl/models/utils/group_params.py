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

import re
from collections import defaultdict
from paddle.vision.models import ResNet
from paddle.vision.models.resnet import BottleneckBlock, BasicBlock


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
        assert re_exps is not None
        if isinstance(re_exps, (tuple, list)):
            for re_str in re_exps:
                matcher_list.append((group_name, re.compile(re_str)))
        else:
            matcher_list.append((group_name, re.compile(re_exps)))
    param_groups = defaultdict(list)
    other_groups = []
    for name, param in model.named_parameters():
        if param.stop_gradient:
           continue
        flag = 0
        for group_name, matcher in matcher_list:
            res = matcher.match(name)
            if res:
                param_groups[group_name].append(name)
                flag = 1
        if flag == 0:
            other_groups.append(name)
    param_groups['others'] = other_groups
    return param_groups


def group_param_layer_decay(
        model,
        weight_decay=0.05,
        group_matcher=None,
        layer_decay: float = .75,
        no_weight_decay_list=(),
        param_names_group=None
    ):
    # weight_decay value can be None and assigned in the optimizer config,
    # but it has the highest priority if given here.
    assert (not group_matcher) or (not param_names_group), \
        "group_matcher and param_names_group should not be given in the same time."
    if group_matcher:
        param_names_group = group_with_matcher(model, group_matcher)
    group_by_name = {z: k for k, v in param_names_group.items() for z in v}
    num_layers = len(param_names_group)
    layer_scales = {z: layer_decay ** (num_layers - i) for i, (k, v) in enumerate(param_names_group.items()) for z in v}
    param_groups = {}
    for name, param in model.named_parameters():
        if param.stop_gradient:
            continue
        lr_scale = layer_scales[name] if name in layer_scales else 1.
        if param.ndim == 1 or any(nd in name for nd in no_weight_decay_list):
            this_decay = 0.
            g_decay = "no_weight_decay"
        else:
            this_decay = weight_decay
            g_decay = "weight_decay"
        new_group_name = group_by_name[name] + '_' + g_decay
        if new_group_name not in param_groups:
            param_groups[new_group_name] = {
                "lr_scale": lr_scale,
                "params": [],
                "group_name": new_group_name,
            }
            if this_decay is not None:
                param_groups[new_group_name]["weight_decay"] = this_decay
        param_groups[new_group_name]["params"].append(param)

    return list(param_groups.values())


def param_group_weight_decay(
        model,
        no_weight_decay_list=(),
        weight_decay=1e-5,
        group_matcher=None):
    # weight_decay value can be None and assigned in the optimizer config,
    # but it has the highest priority if given here.
    param_groups = {}
    if group_matcher is not None:
        param_group_map = group_with_matcher(model, group_matcher)
        group_by_name = {z: k for k, v in param_group_map.items() for z in v}
        for name, param in model.named_parameters():
            if param.stop_gradient:
                continue
            if param.ndim == 1 or any(nd in name for nd in no_weight_decay_list):
                g_decay = "no_weight_decay"
                this_decay = 0.
            else:
                g_decay = "weight_decay"
                this_decay = weight_decay
            new_group_name = group_by_name[name] + "_" + g_decay
            if new_group_name not in param_groups:
                param_groups[new_group_name] = {
                    "params": [],
                    "group_name": new_group_name,
                }
            if this_decay is not None:
                param_groups[new_group_name]["weight_decay"] = this_decay
            param_groups[new_group_name]["params"].append(param)

    else:
        # default
        for name, param in model.named_parameters():
            if param.stop_gradient:
                continue
            if any(nd in name for nd in no_weight_decay_list):
                this_decay = 0.0
                g_decay = "no_weight_decay"
            else:
                g_decay = "weight_decay"
                this_decay = weight_decay
            if g_decay not in param_groups:
                param_groups[g_decay] = {
                    "params": [],
                }
            param_groups[g_decay]["params"].append(param)
            if this_decay is not None:
                param_groups[g_decay]["weight_decay"] = this_decay
    return list(param_groups.values())
