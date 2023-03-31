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

import paddle
import paddle.nn as nn
from .op_count import *


def flops(model, input_size, per_op=True):
    """Get FLOPs and Params

    Args:
      model: Models to be tested.
      input_size: Input image size.
      per_op: Whether to print FLOPs and Params for each op line by line.
    """

    handler_collection = []
    types_collection = set()
    inputs = paddle.randn([1, 3, input_size, input_size])

    def add_hooks(m):
        if len(list(m.children())) > 0:
            return
        m.register_buffer('flops', paddle.zeros([1], dtype='int64'))
        m.register_buffer('params', paddle.zeros([1], dtype='int64'))
        m_type = type(m)

        flops_fn = None
        if m_type in register_hooks:
            flops_fn = register_hooks[m_type]
        else:
            if m_type not in types_collection:
                print(
                    "Cannot find suitable count function for {}. Treat it as zero FLOPs."
                    .format(m_type))

        if flops_fn is not None:
            flops_handler = m.register_forward_post_hook(flops_fn)
            handler_collection.append(flops_handler)

        params_handler = m.register_forward_post_hook(count_parameters)
        io_handler = m.register_forward_post_hook(count_io_info)
        handler_collection.append(params_handler)
        handler_collection.append(io_handler)
        types_collection.add(m_type)

    model.eval()
    model.apply(add_hooks)

    with paddle.no_grad():
        model(inputs)

    total_flops = 0
    total_params = 0
    for m in model.sublayers():
        if len(list(m.children())) > 0:
            continue
        if set(['flops', 'params', 'input_shape',
                'output_shape']).issubset(set(list(m._buffers.keys()))):
            total_flops += m.flops
            total_params += m.params

    table = Table(
        ["Layer Name", "Input Shape", "Output Shape", "Params", "FLOPs"])

    for n, m in model.named_sublayers():
        if len(list(m.children())) > 0:
            continue

        if set(['flops', 'params', 'input_shape',
                'output_shape']).issubset(set(list(m._buffers.keys()))):
            table.add_row([
                m.full_name(),
                list(m.input_shape.numpy()),
                list(m.output_shape.numpy()),
                int(m.params),
                int(m.flops)
            ])
            m._buffers.pop("flops")
            m._buffers.pop("params")
            m._buffers.pop('input_shape')
            m._buffers.pop('output_shape')
    if per_op:
        table.print_table()

    print(f"Total FLOPs : {int(total_flops):,}")
    print(f"Total Params: {int(total_params):,}")


class Table(object):
    """Visual table
    from https://github.com/PaddlePaddle/Paddle/python/paddle/hapi/static_flops.py
    """
    def __init__(self, table_heads):
        self.table_heads = table_heads
        self.table_len = []
        self.data = []
        self.col_num = len(table_heads)
        for head in table_heads:
            self.table_len.append(len(head))

    def add_row(self, row_str):
        if not isinstance(row_str, list):
            print('The row_str should be a list')
        if len(row_str) != self.col_num:
            print(
                'The length of row data should be equal the length of table heads, but the data: {} is not equal table heads {}'
                .format(len(row_str), self.col_num))
        for i in range(self.col_num):
            if len(str(row_str[i])) > self.table_len[i]:
                self.table_len[i] = len(str(row_str[i]))
        self.data.append(row_str)

    def print_row(self, row):
        string = ''
        for i in range(self.col_num):
            string += '|' + str(row[i]).center(self.table_len[i] + 2)
        string += '|'
        print(string)

    def print_shelf(self):
        string = ''
        for length in self.table_len:
            string += '+'
            string += '-' * (length + 2)
        string += '+'
        print(string)

    def print_table(self):
        self.print_shelf()
        self.print_row(self.table_heads)
        self.print_shelf()
        for data in self.data:
            self.print_row(data)
        self.print_shelf()
