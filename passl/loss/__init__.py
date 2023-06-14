#copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

import copy

import paddle
import paddle.nn as nn

from passl.utils import logger
from .celoss import CELoss, ViTCELoss


class CombinedLoss(nn.Layer):
    def __init__(self, config_list):
        super().__init__()
        self.loss_func = []
        self.loss_weight = []
        assert isinstance(config_list, list), (
            'operator config should be a list')
        for config in config_list:
            assert isinstance(config,
                              dict) and len(config) == 1, "yaml format error"
            name = list(config)[0]
            param = config[name]
            assert "weight" in param, "weight must be in param, but param just contains {}".format(
                param.keys())
            self.loss_weight.append(param.pop("weight"))
            self.loss_func.append(eval(name)(**param))

    def __call__(self, input, target):
        if isinstance(input, dict) and (input["logits"].dtype == paddle.float16 or input["logits"].dtype == paddle.bfloat16):
            input["logits"] = paddle.cast(input["logits"], 'float32')
        elif input.dtype == paddle.float16 or input.dtype == paddle.bfloat16:
            input = paddle.cast(input, 'float32')

        loss_dict = {}
        for idx, loss_func in enumerate(self.loss_func):
            loss = loss_func(input, target)
            weight = self.loss_weight[idx]
            loss = {key: loss[key] * weight for key in loss}
            loss_dict.update(loss)
        loss_dict["loss"] = paddle.add_n(list(loss_dict.values()))
        return loss_dict


def build_loss(config):
    module_class = CombinedLoss(copy.deepcopy(config))
    logger.debug("build loss {} success.".format(module_class))
    return module_class
