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
import time
from paddle.autograd import PyLayer
from paddle.fluid import core
from paddle.nn import functional as F

import paddle.distributed as dist
from paddle.distributed.fleet.base import topology as tp
from paddle.distributed.fleet.layers.mpu.random import get_rng_state_tracker

import passl.distributed.distributed_env as dist_env

__all__ = [
    "finer_grained_rowsharded_linear",
    "finer_grained_columnsharded_linear",
    "finer_grained_row_parallel_linear",
    "finer_grained_column_parallel_linear",
    "FinerGrainedRowParallelLinear",
    "FinerGrainedColumnParallelLinear"
]


def finer_grained_rowsharded_linear(x, weight, bias=None, name=None):
    """
    y = x * weight + b = matmul(x, weight) + b
    """

    mp_rank = dist_env.get_model_parallel_world_rank()
    mp_ranks = dist_env.get_model_parallel_world_size()
    p2p_mp_group = dist_env.get_p2p_model_parallel_group()
    mp_group = dist_env.get_model_parallel_group()
    next_mp_rank = (mp_rank + 1) % len(mp_group.ranks)
    prev_mp_rank = (mp_rank - 1 + len(mp_group.ranks)) % len(mp_group.ranks)
    send_group = p2p_mp_group[f'mp_{mp_rank}to{next_mp_rank}']
    recv_group = p2p_mp_group[f'mp_{prev_mp_rank}to{mp_rank}']
    send_dst = mp_group.ranks[next_mp_rank]
    recv_src = mp_group.ranks[prev_mp_rank]

    hidden_size = x.shape[-1]
    assert hidden_size % mp_ranks == 0, f"hidden_size {hidden_size} must be divided by mp_ranks {mp_ranks}"
    micro_hidden_size = hidden_size // mp_ranks
    assert micro_hidden_size == weight.shape[0], f"micro_hidden_size {micro_hidden_size} must be equal to weight.shape[0] {weight.shape[0]}"

    # reverse order [mp_ranks-1, ..., 1, 0]
    cal_index = list(range(mp_ranks-1, -1, -1))
    # shift
    shift = mp_ranks - mp_rank - 1
    cal_index = cal_index[shift:] + cal_index[:shift]

    wi = weight
    y = None

    for idx, t in enumerate(cal_index):
        start = t * micro_hidden_size
        end = start + micro_hidden_size

        # launch async send and recv
        if idx < mp_ranks-1:
            if mp_rank % 2 == 0:
                task_send = dist.isend(wi, dst=send_dst, group=send_group)
            else:
                w_recv = paddle.zeros_like(wi)
                task_recv = dist.irecv(w_recv, src=recv_src, group=recv_group)

            if mp_rank % 2 == 0:
                w_recv = paddle.zeros_like(wi)
                task_recv = dist.irecv(w_recv, src=recv_src, group=recv_group)
            else:
                task_send = dist.isend(wi, dst=send_dst, group=send_group)

        # slice and calculate matmul
        xi = paddle.slice(x, axes=[-1], starts=[start], ends=[end])
        yi = paddle.matmul(xi, wi)

        # sum
        if idx == 0:
            y = yi
        else:
            y = y + yi

        # we need to sync and get received xi
        if idx < mp_ranks-1:
            task_send.wait()
            task_recv.wait()
            wi = w_recv

    if bias is not None:
        y = y + bias

    return y


def finer_grained_columnsharded_linear(x, weight, bias=None, name=None):
    """
    y = x * weight + b = matmul(x, weight) + b
    """

    mp_rank = dist_env.get_model_parallel_world_rank()
    mp_ranks = dist_env.get_model_parallel_world_size()
    p2p_mp_group = dist_env.get_p2p_model_parallel_group()
    mp_group = dist_env.get_model_parallel_group()
    next_mp_rank = (mp_rank + 1) % len(mp_group.ranks)
    prev_mp_rank = (mp_rank - 1 + len(mp_group.ranks)) % len(mp_group.ranks)
    send_group = p2p_mp_group[f'mp_{mp_rank}to{next_mp_rank}']
    recv_group = p2p_mp_group[f'mp_{prev_mp_rank}to{mp_rank}']
    send_dst = mp_group.ranks[next_mp_rank]
    recv_src = mp_group.ranks[prev_mp_rank]

    wi = weight
    y = []

    for idx in range(mp_ranks):
        # launch async send and recv
        if idx < mp_ranks-1:
            if mp_rank % 2 == 0:
                task_send = dist.isend(wi, dst=send_dst, group=send_group)
            else:
                w_recv = paddle.zeros_like(wi)
                task_recv = dist.irecv(w_recv, src=recv_src, group=recv_group)

            if mp_rank % 2 == 0:
                w_recv = paddle.zeros_like(wi)
                task_recv = dist.irecv(w_recv, src=recv_src, group=recv_group)
            else:
                task_send = dist.isend(wi, dst=send_dst, group=send_group)

        # slice and calculate matmul
        yi = paddle.matmul(x, wi)

        y.append(yi)

        # we need to sync and get received xi
        if idx < mp_ranks-1:
            task_send.wait()
            task_recv.wait()
            wi = w_recv

    # shift results
    shift = mp_rank + 1
    y = y[shift:] + y[:shift]
    y = y[::-1]
    y = paddle.concat(y, axis=-1)

    if bias is not None:
        y = y + bias

    return y


class FinerGrainedRowShardedLinearFunction(PyLayer):
    @staticmethod
    def forward(ctx, x, weight, bias=None):
        ctx.save_for_backward([x, weight, bias])
        y = finer_grained_rowsharded_linear(x, weight, bias)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        # TODO(GuoxiaWang): implement backward logic
        raise NotImplementedError(
            "The backward logic is not supported now."
        )
        return grad_output


class FinerGrainedColumnShardedLinearFunction(PyLayer):
    @staticmethod
    def forward(ctx, x, weight, bias=None):
        ctx.save_for_backward([x, weight, bias])
        y = finer_grained_columnsharded_linear(x, weight, bias)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        # TODO(GuoxiaWang): implement backward logic
        raise NotImplementedError(
            "The backward logic is not supported now."
        )
        return grad_output


def finer_grained_row_parallel_linear(x, weight, bias=None, name=None):
    return FinerGrainedRowShardedLinearFunction.apply(x, weight, bias=bias)


def finer_grained_column_parallel_linear(x, weight, bias=None, name=None):
    return FinerGrainedColumnShardedLinearFunction.apply(x, weight, bias=bias)


class FinerGrainedRowParallelLinear(paddle.nn.Layer):
    def __init__(
        self,
        in_features,
        out_features,
        weight_attr=None,
        bias_attr=None,
        input_is_parallel=False,
        gather_output=True,
        mp_group=None,
        name=None,
    ):
        super().__init__()

        self.model_parallel_group = (
            tp._HYBRID_PARALLEL_GROUP.get_model_parallel_group()
            if mp_group is None
            else mp_group
        )
        self.world_size = (
            tp._HYBRID_PARALLEL_GROUP.get_model_parallel_world_size()
            if mp_group is None
            else mp_group.nranks
        )
        self._name = name
        self.is_mp = self.world_size > 1
        self.mp_rank = dist_env.get_model_parallel_world_rank()
        self.input_is_parallel = input_is_parallel
        self.gather_output = gather_output

        assert in_features % self.world_size == 0, (
            "Number of rows of the weight for linear ({}) must be"
            " divisible by model parallel size ({})".format(
                in_features, self.world_size
            )
        )

        self.input_size_per_partition = in_features // self.world_size

        self._weight_attr = weight_attr
        self._dtype = self._helper.get_default_dtype()

        if self.is_mp and paddle.in_dynamic_mode():
            with get_rng_state_tracker().rng_state():
                self.weight = self.create_parameter(
                    shape=[self.input_size_per_partition , out_features],
                    attr=self._weight_attr,
                    dtype=self._dtype,
                    is_bias=False,
                )
        else:
            self.weight = self.create_parameter(
                shape=[self.input_size_per_partition , out_features],
                attr=self._weight_attr,
                dtype=self._dtype,
                is_bias=False,
            )

        self.weight.is_distributed = True if self.is_mp else False

        if self.weight.is_distributed:
            self.weight.split_axis = 0


        self.bias = self.create_parameter(
            shape=[out_features],
            attr=bias_attr,
            dtype=self._dtype,
            is_bias=True,
        )

        self.linear = finer_grained_row_parallel_linear


    def forward(self, x):

        if self.input_is_parallel or (not self.is_mp):
            input_parallel = x
        else:
            input_parallel = paddle.split(x,  self.world_size, axis=0)[self.mp_rank]

        output = self.linear(
            input_parallel, self.weight, self.bias, name=self._name
        )

        if self.is_mp and self.gather_output:
            tensor_list = []
            dist.all_gather(tensor_list, output, group=self.model_parallel_group)
            output = paddle.concat(tensor_list, axis=0)
        return output


class FinerGrainedColumnParallelLinear(paddle.nn.Layer):
    def __init__(
        self,
        in_features,
        out_features,
        weight_attr=None,
        bias_attr=None,
        input_is_parallel=False,
        gather_output=True,
        mp_group=None,
        name=None,
    ):
        super().__init__()

        self.model_parallel_group = (
            tp._HYBRID_PARALLEL_GROUP.get_model_parallel_group()
            if mp_group is None
            else mp_group
        )
        self.world_size = (
            tp._HYBRID_PARALLEL_GROUP.get_model_parallel_world_size()
            if mp_group is None
            else mp_group.nranks
        )
        self._name = name
        self.is_mp = self.world_size > 1
        self.mp_rank = dist_env.get_model_parallel_world_rank()
        self.input_is_parallel = input_is_parallel
        self.gather_output = gather_output
        assert out_features % self.world_size == 0, (
            "Number of column of the weight for linear ({}) must be"
            " divisible by model parallel size ({})".format(
                out_features, self.world_size
            )
        )
        self.output_size_per_partition = out_features // self.world_size

        self._weight_attr = weight_attr
        self._dtype = self._helper.get_default_dtype()

        if self.is_mp and paddle.in_dynamic_mode():
            with get_rng_state_tracker().rng_state():
                self.weight = self.create_parameter(
                    shape=[in_features , self.output_size_per_partition],
                    attr=self._weight_attr,
                    dtype=self._dtype,
                    is_bias=False,
                )
        else:
            self.weight = self.create_parameter(
                shape=[in_features , self.output_size_per_partition],
                attr=self._weight_attr,
                dtype=self._dtype,
                is_bias=False,
            )

        self.weight.is_distributed = True if self.is_mp else False

        if self.weight.is_distributed:
            self.weight.split_axis = 1

        self.bias = self.create_parameter(
            shape=[out_features],
            attr=bias_attr,
            dtype=self._dtype,
            is_bias=True,
        )

        self.linear = finer_grained_column_parallel_linear

    def forward(self, x):
        if self.input_is_parallel or (not self.is_mp):
            input_parallel = x
        else:
            input_parallel = paddle.split(x,  self.world_size, axis=0)[self.mp_rank]

        output = self.linear(
            input_parallel, self.weight, self.bias, name=self._name
        )
        if self.is_mp and self.gather_output:
            tensor_list = []
            dist.all_gather(tensor_list, output, group=self.model_parallel_group)
            output = paddle.concat(tensor_list, axis=0)
        return output
