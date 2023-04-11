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

import math

from abc import abstractmethod
from copy import deepcopy

import paddle
import paddle.nn as nn

from passl.utils import logger
from passl.utils.infohub import runtime_info_hub

class BaseAveragedModel(nn.Layer):
    """A base class for averaging model weights.
    The code is referenced from: https://github.com/open-mmlab/mmengine/blob/main/mmengine/model/averaged_model.py.
    """

    def __init__(self,
                 model,
                 interval: int = 1,
                 update_buffers: bool = False) -> None:
        super().__init__()
        self.model = deepcopy(model)
        for p in self.model.parameters():
            p.stop_gradient = True
        self.interval = interval
        self.register_buffer('steps', paddle.to_tensor(0, dtype=paddle.int64))
        self.update_buffers = update_buffers
        if update_buffers:
            self.avg_parameters = self.model.state_dict()
        else:
            self.avg_parameters = dict(self.model.named_parameters())

    @abstractmethod
    def avg_func(self, averaged_param: paddle.Tensor, source_param: paddle.Tensor,
                 steps: int) -> None:
        """Use in-place operation to compute the average of the parameters. All
        subclasses must implement this method.
        Args:
            averaged_param (Tensor): The averaged parameters.
            source_param (Tensor): The source parameters.
            steps (int): The number of times the parameters have been
                updated.
        """

    def forward(self, *args, **kwargs):
        """Forward method of the averaged model."""
        return self.model(*args, **kwargs)

    def update_parameters(self, model) -> None:
        """Update the parameters of the model. This method will execute the
        ``avg_func`` to compute the new parameters and update the model's
        parameters.
        Args:
            model (nn.Module): The model whose parameters will be averaged.
        """
        src_parameters = (
            model.state_dict()
            if self.update_buffers else dict(model.named_parameters()))
        if self.steps == 0:
            for k, p_avg in self.avg_parameters.items():
                p_avg.copy_(src_parameters[k], False)
        elif self.steps % self.interval == 0:
            for k, p_avg in self.avg_parameters.items():
                if paddle.is_floating_point(p_avg):
                    self.avg_func(p_avg, src_parameters[k], self.steps)
        if not self.update_buffers:
            # If not update the buffers,
            # keep the buffers in sync with the source model.
            for b_avg, b_src in zip(self.model.buffers(), model.buffers()):
                b_avg.copy_(b_src, False)
        self.steps += 1
        
        
class ExponentialMovingAverage(BaseAveragedModel):
    def __init__(self,
                 model,
                 momentum: float = 0.0002,
                 interval: int = 1,
                 update_buffers: bool = False) -> None:
        super().__init__(model, interval, update_buffers)
        assert 0.0 < momentum < 1.0, 'momentum must be in range (0.0, 1.0)'\
                                     f'but got {momentum}'
        if momentum > 0.5:
            logger.warning(
                'The value of momentum in EMA is usually a small number,'
                'which is different from the conventional notion of '
                f'momentum but got {momentum}. Please make sure the '
                f'value is correct.')
        self.momentum = momentum

    def avg_func(self, averaged_param: paddle.Tensor, source_param: paddle.Tensor,
                 steps: int) -> None:
        """Compute the moving average of the parameters using exponential
        moving average.
        Args:
            averaged_param (Tensor): The averaged parameters.
            source_param (Tensor): The source parameters.
            steps (int): The number of times the parameters have been
                updated.
        """
        with paddle.amp.auto_cast(False):
            averaged_param.copy_(paddle.lerp(averaged_param, source_param, self.momentum), False)
        
        
class CosineEMA(ExponentialMovingAverage):
    r"""CosineEMA is implemented for updating momentum parameter, used in BYOL,
    MoCoV3, etc.
    All parameters are updated by the formula as below:
    .. math::
        X'_{t+1} = (1 - m) * X'_t + m * X_t
    Where :math:`m` the the momentum parameter. And it's updated with cosine
    annealing, including momentum adjustment following:
    .. math::
        m = m_{end} + (m_{end} - m_{start}) * (\cos\frac{k\pi}{K} + 1) / 2
    where :math:`k` is the current step, :math:`K` is the total steps.
    .. note::
        This :attr:`momentum` argument is different from one used in optimizer
        classes and the conventional notion of momentum. Mathematically,
        :math:`X'_{t}` is the moving average and :math:`X_t` is the new
        observed value. The value of momentum is usually a small number,
        allowing observed values to slowly update the ema parameters. See also
        :external:py:class:`torch.nn.BatchNorm2d`.
    Args:
        model (nn.Module): The model to be averaged.
        momentum (float): The start momentum value. Defaults to 0.004.
        end_momentum (float): The end momentum value for cosine annealing.
            Defaults to 0.
        interval (int): Interval between two updates. Defaults to 1.
        device (torch.device, optional): If provided, the averaged model will
            be stored on the :attr:`device`. Defaults to None.
        update_buffers (bool): if True, it will compute running averages for
            both the parameters and the buffers of the model. Defaults to
            False.
    """

    def __init__(self,
                 model,
                 max_steps: int = None,
                 momentum: float = 0.004,
                 end_momentum: float = 0.,
                 interval: int = 1,
                 update_buffers: bool = False) -> None:
        super().__init__(
            model=model,
            momentum=momentum,
            interval=interval,
            update_buffers=update_buffers)
        self.end_momentum = end_momentum
        self.max_steps = max_steps

    def avg_func(self, averaged_param: paddle.Tensor,
                 source_param: paddle.Tensor, steps: int) -> None:
        """Compute the moving average of the parameters using the cosine
        momentum strategy.
        Args:
            averaged_param (Tensor): The averaged parameters.
            source_param (Tensor): The source parameters.
            steps (int): The number of times the parameters have been
                updated.
        Returns:
            Tensor: The averaged parameters.
        """
        
        max_steps = self.max_steps
        if max_steps is None:
            max_steps = runtime_info_hub.max_steps
        
        cosine_annealing = (math.cos(math.pi * steps / float(max_steps)) + 1) / 2
        momentum = self.end_momentum - (self.end_momentum -
                                        self.momentum) * cosine_annealing        
        with paddle.amp.auto_cast(False):
            averaged_param.copy_(averaged_param * (1.0 - momentum) + source_param * momentum, False)