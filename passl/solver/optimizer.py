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

import copy
import paddle
from paddle.fluid import framework

from .builder import OPTIMIZERS

OPTIMIZERS.register(paddle.optimizer.Adam)
OPTIMIZERS.register(paddle.optimizer.AdamW)
OPTIMIZERS.register(paddle.optimizer.SGD)
OPTIMIZERS.register(paddle.optimizer.Momentum)
OPTIMIZERS.register(paddle.fluid.optimizer.LarsMomentum)
OPTIMIZERS.register(paddle.optimizer.RMSProp)

@OPTIMIZERS.register()
class PixProLarsMomentum(paddle.fluid.optimizer.LarsMomentum):
    def _append_optimize_op(self, block, param_and_grad):
        assert isinstance(block, framework.Block)
        _lars_weight_decay = self._lars_weight_decay
        param_name = param_and_grad[0].name
        if len(param_and_grad[0].shape) == 1:
            _lars_weight_decay = 0.0

        velocity_acc = self._get_accumulator(self._velocity_acc_str,
                                             param_and_grad[0])
        lr = self._create_param_lr(param_and_grad)

        find_master = self._multi_precision and param_and_grad[
            0].dtype == core.VarDesc.VarType.FP16
        master_weight = (self._master_weights[param_and_grad[0].name]
                         if find_master else None)

        attrs = {
            "mu": self._momentum,
            "lars_coeff": self._lars_coeff,
            "lars_weight_decay": [_lars_weight_decay],
            "multi_precision": find_master,
            "epsilon": self._epsilon,
            "rescale_grad": self._rescale_grad
        }

        inputs = {
            "Param": param_and_grad[0],
            "Grad": param_and_grad[1],
            "Velocity": velocity_acc,
            "LearningRate": lr
        }

        outputs = {"ParamOut": param_and_grad[0], "VelocityOut": velocity_acc}

        if find_master:
            inputs["MasterParam"] = master_weight
            outputs["MasterParamOut"] = master_weight

        # create the momentum optimize op
        momentum_op = block.append_op(
            type=self.type if _lars_weight_decay != 0.0 else 'momentum',
            inputs=inputs,
            outputs=outputs,
            attrs=attrs,
            stop_gradient=True)

        return momentum_op

