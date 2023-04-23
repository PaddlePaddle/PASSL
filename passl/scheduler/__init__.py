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

import paddle

from passl.utils import logger

from .lr_scheduler import TimmCosine, ViTLRScheduler, Step, Poly
from .lr_callable import LRCallable, CosineWithFixLR


def build_lr_scheduler(lr_config, epochs, step_each_epoch):
    lr_config.update({'epochs': epochs, 'step_each_epoch': step_each_epoch})
    if 'name' in lr_config:
        lr_name = lr_config.pop('name')
        lr = eval(lr_name)(**lr_config)
        if isinstance(lr, paddle.optimizer.lr.LRScheduler):
            return lr
        elif isinstance(lr, LRCallable):
            return lr
        else:
            return lr()
    else:
        lr = lr_config['learning_rate']
    logger.debug("build lr ({}) success..".format(lr))
    return lr
