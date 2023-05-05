# copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
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
from paddle import _legacy_C_ops as _C_ops
from .optimizer import Optimizer
from passl.utils import logger


class Momentum(Optimizer):
    def __init__(self,
                 params,
                 lr=0.001,
                 lr_func=None,
                 momentum=0.9,
                 weight_decay=0.0,
                 use_master_param=True,
                 grad_clip=None,
                 **args):

        defaults = dict(
            lr=lr,
            lr_func=lr_func,
            momentum=momentum,
            weight_decay=weight_decay,
            use_master_param=use_master_param,
            grad_clip=grad_clip,
            **args)
        super(Momentum, self).__init__(params, defaults)

    @paddle.no_grad()
    def clear_grad(self, set_to_zero=True):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    p.clear_gradient(set_to_zero)

                if getattr(p, 'has_sparse_grad', None):
                    p.clear_gradient(set_to_zero=False)
                    delattr(p, 'index')
                    delattr(p, 'axis')

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
                        'Momentum does not support sparse gradients.')

                lr = self._get_lr(group)

                state = self.state[p.name]

                # State initialization
                initialized = True
                if len(state) == 0:
                    # Exponential moving average of gradient values
                    state['exp_avg'] = paddle.zeros_like(p)
                    initialized = False

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

                if getattr(p, 'has_sparse_grad', None):
                    index = getattr(p, 'index', None)
                    axis = getattr(p, 'axis', None)
                    assert axis == 0, 'Only support axis=0 now!'
                    assert index is not None
                    assert axis is not None
                    sub_p = paddle.gather(p, index, axis=axis)
                    sub_exp_avg = paddle.gather(exp_avg, index, axis=axis)

                    if group['weight_decay'] != 0.0:
                        grad = (grad + group['weight_decay'] * sub_p
                                ).astype(grad.dtype)

                    if initialized is False:
                        sub_exp_avg.copy_(grad, False)
                    else:
                        sub_exp_avg.copy_(sub_exp_avg * momentum + grad, False)
                    sub_p.copy_(sub_p - lr * sub_exp_avg, False)

                    p.scatter_(index, sub_p)
                    exp_avg.scatter_(index, sub_exp_avg)

                    # _, _, _ = _C_ops.sparse_momentum(
                    #     p,
                    #     grad,
                    #     exp_avg,
                    #     index,
                    #     paddle.to_tensor(
                    #         lr, dtype='float32'),
                    #     master_param,
                    #     p,
                    #     exp_avg,
                    #     master_param,
                    #     'mu',
                    #     momentum,
                    #     'use_nesterov',
                    #     False,
                    #     'regularization_method',
                    #     'l2_decay',
                    #     'regularization_coeff',
                    #     group['weight_decay'],
                    #     'axis',
                    #     axis,
                    #     'multi_precision',
                    #     master_param is not None)
                else:
                    p_fp32 = p
                    if group['use_master_param'] and p.dtype in {
                            paddle.float16, paddle.bfloat16
                    }:
                        p_fp32 = state['master_param']

                    if group['weight_decay'] != 0.0:
                        grad = (grad + group['weight_decay'] * p_fp32
                                ).astype(grad.dtype)

                    if initialized is False:
                        exp_avg.copy_(grad, False)
                    else:
                        exp_avg.copy_(exp_avg * momentum + grad, False)
                    p_fp32.copy_(p_fp32 - lr * exp_avg, False)

                    if p.dtype in {paddle.float16, paddle.bfloat16}:
                        p.copy_(paddle.cast(p_fp32, p.dtype), False)

                    # _, _, _ = _C_ops.momentum(
                    #     p,
                    #     grad,
                    #     exp_avg,
                    #     paddle.to_tensor(
                    #         lr, dtype='float32'),
                    #     master_param,
                    #     p,
                    #     exp_avg,
                    #     master_param,
                    #     'mu',
                    #     momentum,
                    #     'use_nesterov',
                    #     False,
                    #     'regularization_method',
                    #     'l2_decay',
                    #     'regularization_coeff',
                    #     group['weight_decay'],
                    #     'multi_precision',
                    #     master_param is not None)
