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


class AdamW(Optimizer):
    def __init__(self,
                 params,
                 lr=0.001,
                 lr_func=None,
                 betas=(0.9, 0.999),
                 eps=1e-8,
                 weight_decay=0.0,
                 use_master_param=False,
                 exp_avg_force_fp32=False,
                 grad_clip=None,
                 **args):

        defaults = dict(
            lr=lr,
            lr_func=lr_func,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            use_master_param=use_master_param,
            exp_avg_force_fp32=exp_avg_force_fp32,
            grad_clip=grad_clip,
            **args)
        super(AdamW, self).__init__(params, defaults)

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
                        'Adafactor does not support sparse gradients.')

                lr = self._get_lr(group)

                state = self.state[p.name]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    dtype = p.dtype
                    if group['exp_avg_force_fp32']:
                        dtype = 'float32'
                    # Exponential moving average of gradient values
                    state['exp_avg'] = paddle.zeros_like(p, dtype=dtype)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = paddle.zeros_like(p, dtype='float32')

                    if group['use_master_param'] and p.dtype in {
                            paddle.float16, paddle.bfloat16
                    }:
                        state['master_param'] = paddle.cast(p, dtype='float32')

                state['step'] += 1
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                beta1_pow = paddle.to_tensor(beta1**state['step'])
                beta2_pow = paddle.to_tensor(beta2**state['step'])

                with_decay = False
                if group['weight_decay'] != 0.0:
                    with_decay = True

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
                    sub_exp_avg_sq = paddle.gather(
                        exp_avg_sq, index, axis=axis)

                    _, _, _, _, _, _ = _C_ops.adamw(
                        sub_p, grad,
                        paddle.to_tensor(lr), sub_exp_avg, sub_exp_avg_sq,
                        beta1_pow, beta2_pow, master_param, sub_p, sub_exp_avg,
                        sub_exp_avg_sq, beta1_pow, beta2_pow, master_param,
                        'epsilon', group['eps'], 'lazy_mode', False,
                        'min_row_size_to_use_multithread', 1000, 'beta1',
                        beta1, 'beta2', beta2, "with_decay", with_decay,
                        'coeff', group['weight_decay'], 'multi_precision',
                        master_param is not None, 'lr_ratio', 1.0)

                    p.scatter_(index, sub_p)
                    exp_avg.scatter_(index, sub_exp_avg)
                    exp_avg_sq.scatter_(index, sub_exp_avg_sq)

                else:
                    _, _, _, _, _, _ = _C_ops.adamw(
                        p, grad,
                        paddle.to_tensor(lr), exp_avg, exp_avg_sq, beta1_pow,
                        beta2_pow, master_param, p, exp_avg, exp_avg_sq,
                        beta1_pow, beta2_pow, master_param, 'epsilon',
                        group['eps'], 'lazy_mode', False,
                        'min_row_size_to_use_multithread', 1000, 'beta1',
                        beta1, 'beta2', beta2, "with_decay", with_decay,
                        'coeff', group['weight_decay'], 'multi_precision',
                        master_param is not None, 'lr_ratio', 1.0)
