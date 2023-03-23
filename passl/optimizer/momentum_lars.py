# copyright (c) 2022 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import paddle
from .optimizer import Optimizer
from passl.utils import logger


class MomentumLARS(Optimizer):
    """
    LARS optimizer, implementation from MoCo v3:
    https://github.com/facebookresearch/moco-v3

    No rate scaling or weight decay for parameters <= 1D.
    """

    def __init__(self,
                 params,
                 lr=0.0,
                 lr_func=None,
                 momentum=0.9,
                 weight_decay=0.0,
                 trust_coefficient=0.001,
                 use_master_param=True,
                 grad_clip=None,
                 **args):

        defaults = dict(
            lr=lr,
            lr_func=lr_func,
            momentum=momentum,
            weight_decay=weight_decay,
            trust_coefficient=trust_coefficient,
            use_master_param=use_master_param,
            grad_clip=grad_clip,
            **args)
        super(MomentumLARS, self).__init__(params, defaults)

    @staticmethod
    def _get_lr(param_group):
        lr_t = param_group["lr"]
        if isinstance(lr_t, paddle.optimizer.lr.LRScheduler):
            lr_t = lr_t.get_lr()
        return lr_t

    @paddle.no_grad()
    def step(self):
        for group in self.param_groups:
            if group['grad_clip'] is not None:
                group['grad_clip'](group['params'])
            for p in group['params']:
                grad = p.grad
                if grad is None:
                    continue

                if grad.is_selected_rows():
                    raise RuntimeError(
                        'MomentumLARS does not support sparse gradients.')

                lr = self._get_lr(group)

                state = self.state[p.name]

                # State initialization
                if len(state) == 0:
                    # Exponential moving average of gradient values
                    state['exp_avg'] = paddle.zeros_like(p)

                    if group['use_master_param'] and p.dtype in {
                            paddle.float16, paddle.bfloat16
                    }:
                        state['master_param'] = paddle.cast(p, dtype='float32')

                exp_avg = state['exp_avg']
                momentum = group['momentum']

                master_param = None
                if group['use_master_param'] and p.dtype in {
                        paddle.float16, paddle.bfloat16
                }:
                    master_param = state['master_param']

                p_fp32 = p
                if group['use_master_param'] and p.dtype in {
                        paddle.float16, paddle.bfloat16
                }:
                    p_fp32 = state['master_param']

                if p_fp32.ndim > 1:  # if not normalization gamma/beta or bias
                    grad = (grad + group['weight_decay'] * p_fp32
                            ).astype(grad.dtype)
                    param_norm = paddle.norm(p_fp32)
                    update_norm = paddle.norm(grad)
                    one = paddle.ones_like(param_norm)
                    q = paddle.where(param_norm > 0.,
                                     paddle.where(update_norm > 0, (
                                         group['trust_coefficient'] *
                                         param_norm / update_norm), one), one)
                    grad = grad.multiply(q)

                if group['weight_decay'] != 0.0:
                    grad = (grad + group['weight_decay'] * p_fp32
                            ).astype(grad.dtype)

                exp_avg.copy_(exp_avg * momentum + grad, False)
                p_fp32.copy_(p_fp32 - lr * exp_avg, False)

                if p.dtype in {paddle.float16, paddle.bfloat16}:
                    p.copy_(paddle.cast(p_fp32, p.dtype), False)
