# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

from collections import defaultdict
from paddle.amp import GradScaler as FrameworkGradScaler
from paddle.amp import OptimizerState
from paddle import _legacy_C_ops as _C_ops
import paddle


class GradScaler(FrameworkGradScaler):
    def __init__(self,
                 enable=True,
                 init_loss_scaling=2.**10,
                 max_loss_scaling=2.**32,
                 incr_ratio=2.0,
                 decr_ratio=0.5,
                 incr_every_n_steps=1000,
                 decr_every_n_nan_or_inf=2,
                 use_dynamic_loss_scaling=True,
                 no_unscale_list=[]):
        super(GradScaler, self).__init__(enable, init_loss_scaling, incr_ratio,
                                         decr_ratio, incr_every_n_steps,
                                         decr_every_n_nan_or_inf,
                                         use_dynamic_loss_scaling)
        self.max_loss_scaling = paddle.to_tensor(max_loss_scaling, 'float32')
        self._found_inf = paddle.to_tensor(False)
        self.no_unscale_list = no_unscale_list

    @paddle.no_grad()
    def step(self, optimizer):
        if hasattr(self, '_scale'):
            self._scale.clip_(max=self.max_loss_scaling)
        super(GradScaler, self).step(optimizer)

    @paddle.no_grad()
    def _unscale(self, optimizer):
        if not self._enable:
            return

        optimizer_state = self._optimizer_states[id(optimizer)]

        if optimizer_state["state"] is OptimizerState.UNSCALED:
            raise RuntimeError(
                "unscale_() has already been called on this optimizer since the last update()."
            )
        elif optimizer_state["state"] is OptimizerState.STEPPED:
            raise RuntimeError("unscale_() is being called after step().")

        param_grads_fp16 = []
        param_grads_fp32 = []
        for group in optimizer._param_groups:
            for param in group['params']:
                if param._grad_ivar() is not None and not any(
                        name in param.name for name in self.no_unscale_list):
                    if param._grad_ivar().dtype == paddle.float16:
                        param_grads_fp16.append(param._grad_ivar())
                    else:
                        param_grads_fp32.append(param._grad_ivar())

        if len(param_grads_fp16):
            _C_ops.check_finite_and_unscale(param_grads_fp16, self._scale,
                                            param_grads_fp16,
                                            self._temp_found_inf_fp16)
        if len(param_grads_fp32):
            _C_ops.check_finite_and_unscale(param_grads_fp32, self._scale,
                                            param_grads_fp32,
                                            self._temp_found_inf_fp32)
        if len(param_grads_fp16) and len(param_grads_fp32):
            self._found_inf = self._temp_found_inf_fp16 or self._temp_found_inf_fp32
        elif len(param_grads_fp16):
            self._found_inf = self._temp_found_inf_fp16
        else:
            self._found_inf = self._temp_found_inf_fp32

        optimizer_state["state"] = OptimizerState.UNSCALED
