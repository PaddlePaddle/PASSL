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

# Ref: https://github.com/openai/DALL-E/blob/master/dall_e/decoder.py

import attr

import paddle
import paddle.nn as nn

from functools import partial
from dall_e.utils import Conv2D


@attr.s(eq=False, repr=False)
class DecoderBlock(nn.Layer):
    n_in: int = attr.ib(validator=lambda i, a, x: x >= 1)
    n_out: int = attr.ib(validator=lambda i, a, x: x >= 1 and x % 4 == 0)
    n_layers: int = attr.ib(validator=lambda i, a, x: x >= 1)

    requires_grad: bool = attr.ib(default=False)

    def __attrs_post_init__(self) -> None:
        super().__init__()
        self.n_hid = self.n_out // 4
        self.post_gain = 1 / (self.n_layers**2)

        make_conv = partial(Conv2D, requires_grad=self.requires_grad)
        self.id_path = make_conv(
            self.n_in, self.n_out,
            1) if self.n_in != self.n_out else nn.Identity()
        self.res_path = nn.Sequential(
            ('relu_1', nn.ReLU()),
            ('conv_1', make_conv(self.n_in, self.n_hid, 1)),
            ('relu_2', nn.ReLU()),
            ('conv_2', make_conv(self.n_hid, self.n_hid, 3)),
            ('relu_3', nn.ReLU()),
            ('conv_3', make_conv(self.n_hid, self.n_hid, 3)),
            ('relu_4', nn.ReLU()),
            ('conv_4', make_conv(self.n_hid, self.n_out, 3)), )

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        return self.id_path(x) + self.post_gain * self.res_path(x)


@attr.s(eq=False, repr=False)
class Decoder(nn.Layer):
    group_count: int = 4
    n_init: int = attr.ib(default=128, validator=lambda i, a, x: x >= 8)
    n_hid: int = attr.ib(default=256, validator=lambda i, a, x: x >= 64)
    n_blk_per_group: int = attr.ib(default=2, validator=lambda i, a, x: x >= 1)
    output_channels: int = attr.ib(default=3, validator=lambda i, a, x: x >= 1)
    vocab_size: int = attr.ib(default=8192, validator=lambda i, a, x: x >= 512)

    requires_grad: bool = attr.ib(default=False)
    use_mixed_precision: bool = attr.ib(default=True)

    def __attrs_post_init__(self) -> None:
        super().__init__()

        blk_range = range(self.n_blk_per_group)
        n_layers = self.group_count * self.n_blk_per_group
        make_conv = partial(Conv2D, requires_grad=self.requires_grad)
        make_blk = partial(
            DecoderBlock, n_layers=n_layers, requires_grad=self.requires_grad)

        self.blocks = nn.Sequential(
            ('input', make_conv(
                self.vocab_size, self.n_init, 1, use_float16=False)),
            ('group_1', nn.Sequential(
                * [(f'block_{i + 1}', make_blk(self.n_init if i == 0 else
                                               8 * self.n_hid, 8 * self.n_hid))
                   for i in blk_range],
                ('upsample', nn.Upsample(
                    scale_factor=2, mode='nearest')), )),
            ('group_2', nn.Sequential(
                * [(f'block_{i + 1}', make_blk(8 * self.n_hid if i == 0 else
                                               4 * self.n_hid, 4 * self.n_hid))
                   for i in blk_range],
                ('upsample', nn.Upsample(
                    scale_factor=2, mode='nearest')), )),
            ('group_3', nn.Sequential(
                * [(f'block_{i + 1}', make_blk(4 * self.n_hid if i == 0 else
                                               2 * self.n_hid, 2 * self.n_hid))
                   for i in blk_range],
                ('upsample', nn.Upsample(
                    scale_factor=2, mode='nearest')), )),
            ('group_4', nn.Sequential(* [(f'block_{i + 1}', make_blk(
                2 * self.n_hid if i == 0 else 1 * self.n_hid, 1 * self.n_hid))
                                         for i in blk_range], )),
            ('output', nn.Sequential(
                ('relu', nn.ReLU()),
                ('conv', make_conv(1 * self.n_hid, 2 * self.output_channels,
                                   1)), )), )

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        if len(x.shape) != 4:
            raise ValueError(f'input shape {x.shape} is not 4d')
        if x.shape[1] != self.vocab_size:
            raise ValueError(
                f'input has {x.shape[1]} channels but model built for {self.vocab_size}'
            )
        if x.dtype != paddle.float32:
            raise ValueError('input must have dtype paddle.float32')

        return self.blocks(x)
