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

import math
import paddle
from .optimizer import Optimizer


class Adafactor(Optimizer):
    def __init__(self,
                 params,
                 lr=None,
                 lr_func=None,
                 eps=1e-30,
                 eps_scale=1e-3,
                 clip_threshold=1.0,
                 decay_rate=-0.8,
                 betas=None,
                 weight_decay=0.0,
                 scale_parameter=True,
                 warmup_init=False,
                 use_master_param=True,
                 no_weight_decay_name=[],
                 one_dim_param_no_weight_decay=False,
                 grad_clip=None,
                 **args):

        relative_step = not lr
        if warmup_init and not relative_step:
            raise ValueError('warmup_init requires relative_step=True')

        beta1 = None if betas is None else betas[
            0]  # make it compat with standard betas arg
        defaults = dict(
            lr=lr,
            lr_func=lr_func,
            eps=eps,
            eps_scale=eps_scale,
            clip_threshold=clip_threshold,
            decay_rate=decay_rate,
            beta1=beta1,
            weight_decay=weight_decay,
            scale_parameter=scale_parameter,
            relative_step=relative_step,
            warmup_init=warmup_init,
            use_master_param=use_master_param,
            no_weight_decay_name=no_weight_decay_name,
            one_dim_param_no_weight_decay=one_dim_param_no_weight_decay,
            grad_clip=grad_clip,
            **args)
        super(Adafactor, self).__init__(params, defaults)

    @staticmethod
    def _get_lr(param_group, param_state):
        lr_t = param_group["lr"]
        if param_group['relative_step']:
            min_step = 1e-6 * param_state['step'] if param_group[
                'warmup_init'] else 1e-2
            lr_t = min(min_step, 1.0 / math.sqrt(param_state['step']))
            if not param_group['scale_parameter']:
                lr_t *= 0.05
        elif isinstance(lr_t, paddle.optimizer.lr.LRScheduler):
            lr_t = lr_t.get_lr()
        param_scale = 1.0
        if param_group['scale_parameter']:
            param_scale = max(param_group['eps_scale'], param_state['RMS'])
        return lr_t * param_scale

    @staticmethod
    def _get_options(param_group, param_shape):
        factored = len(param_shape) >= 2
        use_first_moment = param_group['beta1'] is not None
        return factored, use_first_moment

    @staticmethod
    def _rms(tensor):
        return paddle.norm(tensor, 2) / (tensor.numel()**0.5)

    def _approx_sq_grad(self, exp_avg_sq_row, exp_avg_sq_col):
        r_factor = (exp_avg_sq_row / exp_avg_sq_row.mean(
            axis=-1, keepdim=True)).rsqrt_().unsqueeze(-1)
        c_factor = exp_avg_sq_col.unsqueeze(-2).rsqrt()
        return r_factor * c_factor

    @paddle.no_grad()
    def step(self):
        for group in self.param_groups:
            if group['grad_clip'] is not None:
                group['grad_clip'](group['params'])

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.dtype in {paddle.float16, paddle.bfloat16}:
                    grad = paddle.cast(grad, 'float32')

                if grad.is_selected_rows():
                    raise RuntimeError(
                        'Adafactor does not support sparse gradients.')

                state = self.state[p.name]

                factored, use_first_moment = self._get_options(group,
                                                               grad.shape)
                # State Initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['RMS'] = 0

                    if use_first_moment:
                        # Exponential moving average of gradient values
                        state['exp_avg'] = paddle.zeros_like(grad)
                    if factored:
                        state['exp_avg_sq_row'] = paddle.zeros(grad.shape[:-1])
                        state['exp_avg_sq_col'] = paddle.zeros(
                            grad.shape[:-2] + grad.shape[-1:])
                    else:
                        state['exp_avg_sq'] = paddle.zeros_like(grad)

                    if group['use_master_param'] and p.dtype in {
                            paddle.float16, paddle.bfloat16
                    }:
                        state['master_param'] = paddle.cast(p, dtype='float32')

                p_fp32 = p
                if group['use_master_param'] and p.dtype in {
                        paddle.float16, paddle.bfloat16
                }:
                    p_fp32 = state['master_param']

                state['step'] += 1
                state['RMS'] = self._rms(p_fp32)

                # \alpha_{t}=\max \left(\epsilon_{2}, \operatorname{RMS}\left(X_{t-1}\right)\right) \rho_{t}
                # \rho_{t}=\min \left(10^{-2}, \frac{1}{\sqrt{t}}\right)
                lr_t = self._get_lr(group, state)

                # \hat{\beta}_{2 t}=1-t^{-0.8}
                beta2t = 1.0 - math.pow(state['step'], group['decay_rate'])

                # G_{t}^{2}+\epsilon_{1} 1_{n} 1_{m}^{\top}
                update = grad**2 + group['eps']
                if factored:
                    state['exp_avg_sq_row'] = beta2t * state[
                        'exp_avg_sq_row'] + (1.0 - beta2t) * update.mean(
                            axis=-1)
                    state['exp_avg_sq_col'] = beta2t * state[
                        'exp_avg_sq_col'] + (1.0 - beta2t) * update.mean(
                            axis=-2)

                    # Approximation of exponential moving average of square of gradient
                    update = self._approx_sq_grad(state['exp_avg_sq_row'],
                                                  state['exp_avg_sq_col'])
                    update = update * grad
                else:
                    state['exp_avg_sq'] = beta2t * state['exp_avg_sq'] + (
                        1.0 - beta2t) * update
                    update = paddle.rsqrt(state['exp_avg_sq']) * grad

                update = update / (self._rms(update) /
                                   group['clip_threshold']).clip_(min=1.0)
                update = update * lr_t

                if use_first_moment:
                    state['exp_avg'] = group['beta1'] * state['exp_avg'] + (
                        1.0 - group['beta1']) * update
                    update = state['exp_avg']

                if group['weight_decay'] != 0 and not (
                        any(name in p.name
                            for name in group['no_weight_decay_name']) or
                    (group['one_dim_param_no_weight_decay'] and
                     len(p.shape) == 1)):
                    p_fp32.copy_(p_fp32 * (1.0 - group['weight_decay'] * lr_t),
                                 False)

                p_fp32.copy_(p_fp32 - update, False)
                if p.dtype in {paddle.float16, paddle.bfloat16}:
                    p.copy_(paddle.cast(p_fp32, p.dtype), False)
