# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

from paddle import _C_ops


def l2_normalize(x, axis, epsilon=1e-12, name=None):
    r"""
    This op normalizes `x` along dimension `axis` using an L2
    norm. For a 1-D tensor (`dim` is fixed to 0), this layer computes
    .. math::
        y = \\frac{x}{ \sqrt{\sum {x^2} + epsion }}
    For `x` with more dimensions, this layer independently normalizes each 1-D
    slice along dimension `axis`.
    Args:
        x(Variable|list): The input tensor could be N-D tensor, and the input data type could be float16, float32 or float64.
        axis(int): The axis on which to apply normalization. If `axis < 0`, \
            the dimension to normalization is rank(X) + axis. -1 is the
            last dimension.
        epsilon(float): The epsilon value is used to avoid division by zero, \
            the default value is 1e-12.
    name(str, optional): The default value is None.  Normally there is no need for user to set this property.  For more information, please refer to :ref:`api_guide_Name`
    Returns:
        Variable: The output has the same shape and data type with `x`.
    """
    if len(x.shape) == 1:
        axis = 0
    out, _ = _C_ops.norm(x, 1 if axis is None else axis, epsilon, False)
    return out
