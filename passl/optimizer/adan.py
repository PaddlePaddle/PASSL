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

"""
Code ref from https://github.com/sail-sg/Adan
"""

import math

import paddle

from passl.optimizer import Optimizer
from paddle.optimizer import adam

class Adan(Optimizer):
    """
    Implements a pytorch variant of Adan
    Adan was proposed in
    Adan: Adaptive Nesterov Momentum Algorithm for Faster Optimizing Deep Models[J]. arXiv preprint arXiv:2208.06677, 2022.
    https://arxiv.org/abs/2208.06677
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups.
        lr (float, optional): learning rate. (default: 1e-3)
        betas (Tuple[float, float, float], optional): coefficients used for computing
            running averages of gradient and its norm. (default: (0.98, 0.92, 0.99))
        eps (float, optional): term added to the denominator to improve
            numerical stability. (default: 1e-8)
        weight_decay (float, optional): decoupled weight decay (L2 penalty) (default: 0)
        no_prox (bool): how to perform the decoupled weight decay (default: False)
        grad_clip (ClipGradByGlobalNorm, optional): Gradient cliping strategy (default: None).
        use_master_param (bool, optional): whether to use multi-precision during weight updating (default: False).
        is_vanilla (bool, optional): whether to use vanilla adan (default: False).
    """

    def __init__(
            self,
            params,
            lr=1e-3,
            betas=(0.98, 0.92, 0.99),
            lr_func=None,
            eps=1e-8,
            weight_decay=0.0,
            no_prox=False,
            grad_clip=None,
            use_master_param=False,
            is_vanilla=False,
    ):
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= betas[2] < 1.0:
            raise ValueError("Invalid beta parameter at index 2: {}".format(betas[2]))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, no_prox=no_prox, is_vanilla=is_vanilla, grad_clip=grad_clip,use_master_param=use_master_param)
        super(Adan, self).__init__(params, defaults)

    @paddle.no_grad()
    def step(self, closure=None):
        """ Performs a single optimization step.
        """
        for group in self.param_groups:
            # TODO Make sure if nee
            if group['grad_clip'] is not None:
                group['grad_clip'](group['params'])

            beta1, beta2, beta3 = group['betas']
            # assume same step across group now to simplify things
            # per parameter step can be easily support by making it tensor, or pass list into kernel
            if 'step' in group:
                group['step'] += 1
            else:
                group['step'] = 1

            lr = self._get_lr(group)

            bias_correction1 = 1.0 - beta1 ** group['step']
            bias_correction2 = 1.0 - beta2 ** group['step']
            bias_correction3 = 1.0 - beta3 ** group['step']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.dtype in {paddle.float16, paddle.bfloat16}:
                    grad = paddle.cast(grad, 'float32')
                state = self.state[p]
                if len(state) == 0:
                    state['exp_avg'] = paddle.zeros_like(grad)
                    if not group['is_vanilla']:
                        state['exp_avg_diff'] = paddle.zeros_like(grad)
                    state['exp_avg_sq'] = paddle.zeros_like(grad)
                    state['pre_grad'] = grad.clone()
                    if group['use_master_param'] and p.dtype in {
                            paddle.float16, paddle.bfloat16
                    }:
                        state['master_param'] = paddle.cast(p, dtype='float32')

                master_param = p
                if group['use_master_param'] and p.dtype in {
                        paddle.float16, paddle.bfloat16
                }:
                    master_param = state['master_param']

                grad_diff = grad - state['pre_grad']

                if group['is_vanilla']:
                    state['exp_avg'] = state['exp_avg'] * beta1 + (1.0 - beta1) * grad  + beta2 * (1 - beta2) * grad_diff
                else:
                    state['exp_avg'] = state['exp_avg'] * beta1 + grad * ( 1. - beta1)  # m_t
                    state['exp_avg_diff'] = state['exp_avg_diff'] * beta2 + grad_diff* (1. - beta2)  # diff_t (v)
                update = grad + beta2 * grad_diff
                state['exp_avg_sq'] = state['exp_avg_sq'] * beta3
                state['exp_avg_sq'].add_((1. - beta3)* (update * update))  # n_t

                denom = (state['exp_avg_sq'].sqrt() / math.sqrt(bias_correction3)) + group['eps']
                if group['is_vanilla']:
                    update = (state['exp_avg'] / bias_correction1 + beta2 * grad_diff / bias_correction2).divide(denom)
                else:
                    update = (state['exp_avg'] / bias_correction1 + beta2 * state['exp_avg_diff'] / bias_correction2).divide(denom)

                if group['no_prox']:
                    master_param = master_param * (1 - lr * group['weight_decay'])
                    master_param = master_param - (update * lr)
                    p.copy_(paddle.cast(master_param, p.dtype),False)
                else:
                    master_param = master_param - (update * lr)
                    master_param = master_param / (1 + lr * group['weight_decay'])
                    p.copy_(paddle.cast(master_param, p.dtype),False)

                if group['use_master_param'] and p.dtype in {
                        paddle.float16, paddle.bfloat16
                }:
                    state['master_param'] = master_param

                state['pre_grad'].copy_(grad,False)
