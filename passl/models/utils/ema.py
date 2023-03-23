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

import paddle


class EMA(object):
    """
    Exponential Moving Average.
    """

    def __init__(self, param_groups, decay=0.9999, thres_steps=True):
        self._param_groups = param_groups
        self._decay = decay
        self._thres_steps = thres_steps
        self._shadow = {}
        self._backup = {}
        self._update_step = 0

    @paddle.no_grad()
    def register(self):
        """Register."""

        for group in self._param_groups:
            for p in group['params']:
                if p.stop_gradient is True:
                    continue
                self._shadow[p.name] = p.detach().clone()

    @paddle.no_grad()
    def update(self):
        """Update params."""
        decay = min(self._decay, (1 + self._update_step) / (
            10 + self._update_step)) if self._thres_steps else self._decay

        for group in self._param_groups:
            for p in group['params']:
                if p.stop_gradient is True:
                    continue
                old_val = self._shadow[p.name]
                new_average = decay * old_val + (1 - decay) * p
                self._shadow[p.name] = new_average

        self._update_step += 1
        return decay

    @paddle.no_grad()
    def apply_shadow(self):
        """Apply shadow params."""

        for group in self._param_groups:
            for p in group['params']:
                if p.stop_gradient is True:
                    continue
                assert p.name in self._shadow

                self._backup[p.name] = p.detach().clone()
                p.set_value(self._shadow[p.name])

    @paddle.no_grad()
    def restore(self):
        """Restore params."""

        for group in self._param_groups:
            for p in group['params']:
                if p.stop_gradient is True:
                    continue
                assert p.name in self._backup
                p.set_value(self._backup[p.name])
        self._backup = {}

    @paddle.no_grad()
    def state_dict(self):
        return {
            'shadow': self._shadow,
            'thres_steps': self._thres_steps,
            'update_step': self._update_step,
            'decay': self._decay,
        }

    @paddle.no_grad()
    def set_state_dict(self, state_dict):
        self._shadow = state_dict['shadow']
        self._thres_steps = state_dict['thres_steps']
        self._update_step = state_dict['update_step']
        self._decay = state_dict['decay']
