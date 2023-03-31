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

import warnings
import paddle
from paddle import _legacy_C_ops as _C_ops
from passl.utils import logger


def _squared_l2_norm(x):
    if x.dtype == paddle.float16:
        square = paddle.square(x)
        sum_square = paddle.sum(square)
        return sum_square

    return _C_ops.squared_l2_norm(x)


class ClipGradByGlobalNorm(object):
    def __init__(self,
                 clip_norm=1.0,
                 clip_norm_max=None,
                 always_clip=False,
                 no_clip_list=[]):
        self.clip_norm = clip_norm
        self.clip_norm_max = clip_norm_max
        self.no_clip_list = no_clip_list
        self.always_clip = always_clip

    def __call__(self, params):
        sum_square_list_fp16 = []
        sum_square_list_fp32 = []
        for param in params:
            if param.grad is None or any(name in param.name
                                         for name in self.no_clip_list):
                continue
            if getattr(param, 'need_clip', True) is False:
                continue
            assert param.grad.dtype in [paddle.float32, paddle.float16]
            sum_square = _squared_l2_norm(param.grad)
            if param.grad.dtype == paddle.float32:
                sum_square_list_fp32.append(sum_square)
            elif param.grad.dtype == paddle.float16:
                sum_square_list_fp16.append(sum_square)

        if len(sum_square_list_fp32) <= 0 and len(sum_square_list_fp16) <= 0:
            warnings.warn('grads_fp32 and grads_fp16 are empty')
            return None

        global_norm_var = []
        if len(sum_square_list_fp16) > 0:
            global_norm_var_fp16 = paddle.add_n(sum_square_list_fp16)
            global_norm_var.append(global_norm_var_fp16.astype("float32"))
        if len(sum_square_list_fp32) > 0:
            global_norm_var_fp32 = paddle.add_n(sum_square_list_fp32)
            global_norm_var.append(global_norm_var_fp32)

        global_norm = paddle.add_n(global_norm_var)
        global_norm = paddle.sqrt(global_norm)

        if not self.always_clip and global_norm <= self.clip_norm:
            return

        clip_coef_fp32 = self.clip_norm / (global_norm + 1e-6)
        if self.clip_norm_max is not None:
            clip_coef_fp32 = paddle.clip(
                clip_coef_fp32, max=self.clip_norm_max)

        for param in params:
            if param.grad is None or any(name in param.name
                                         for name in self.no_clip_list):
                continue
            if getattr(param, 'need_clip', True) is False:
                continue

            clip_coef = clip_coef_fp32
            if param.grad.dtype == paddle.float16:
                clip_coef = clip_coef_fp32.astype("float16")

            param.grad.detach().scale_(clip_coef)


@paddle.no_grad()
def clip_grad_norm_(parameters,
                    max_norm: float,
                    norm_type: float=2.0,
                    error_if_nonfinite: bool=False):
    r"""Clips gradient norm of an iterable of parameters.

    The norm is computed over all gradients together, as if they were
    concatenated into a single vector. Gradients are modified in-place.

    Args:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        max_norm (float or int): max norm of the gradients
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.
        error_if_nonfinite (bool): if True, an error is thrown if the total
            norm of the gradients from :attr:``parameters`` is ``nan``,
            ``inf``, or ``-inf``. Default: False (will switch to True in the future)

    Returns:
        Total norm of the parameters (viewed as a single vector).
    """
    if isinstance(parameters, paddle.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return paddle.to_tensor([0.])

    total_norm = paddle.norm(
        paddle.stack([paddle.norm(p.grad, norm_type) for p in parameters]),
        norm_type)
    if error_if_nonfinite and paddle.logical_or(total_norm.isnan(),
                                                total_norm.isinf()):
        raise RuntimeError(
            f'The total norm of order {norm_type} for gradients from '
            '`parameters` is non-finite, so it cannot be clipped. To disable '
            'this error and scale the gradients by the non-finite norm anyway, '
            'set `error_if_nonfinite=False`')
    clip_coef = max_norm / (total_norm + 1e-6)
    clip_coef_clamped = paddle.clip(clip_coef, max=1.0)
    for p in parameters:
        p.grad.detach().scale_(clip_coef_clamped)
    return total_norm
