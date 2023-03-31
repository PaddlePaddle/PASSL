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
import paddle

from passl.core.grad_clip import ClipGradByGlobalNorm
from passl.core.param_fuse import get_fused_params

from passl.utils import logger

from .optimizer import Optimizer
from .adamw import AdamW
from .adafactor import Adafactor
from .momentum import Momentum
from .momentum_lars import MomentumLARS


def build_optimizer(config, lr_scheduler, model=None):
    config = copy.deepcopy(config)

    grad_clip = None
    grad_clip_config = config.pop('grad_clip', None)
    if grad_clip_config is not None:
        grad_clip_name = grad_clip_config.pop('name', 'ClipGradByGlobalNorm')
        grad_clip = eval(grad_clip_name)(**grad_clip_config)

    no_weight_decay_name = config.pop('no_weight_decay_name', [])

    param_group = defaultdict(list)
    for n, p in model.named_parameters():
        state = copy.deepcopy(p.__dict__)
        if any(nd in n for nd in no_weight_decay_name):
            state['no_weight_decay'] = True
        param_group[str(state)].append(p)

    # fuse params
    for key in param_group:
        if 'gpu' not in paddle.get_device():
            continue
        if "'is_distributed': True" in key:
            continue
        if "'has_sparse_grad': True" in key:
            continue

        param_group[key] = get_fused_params(param_group[key])

    # bulid optimizer params
    params = []
    for key in param_group:
        group = {'params': param_group[key]}

        if "'is_distributed': True" in key:
            group['is_distributed'] = True

        if 'no_weight_decay' in key:
            group['weight_decay'] = 0.0

        params.append(group)

    optim_name = config.pop('name')
    optim = eval(optim_name)(params,
                             lr=lr_scheduler,
                             grad_clip=grad_clip,
                             **config)
    logger.debug("build optimizer ({}) success..".format(optim))
    return optim
