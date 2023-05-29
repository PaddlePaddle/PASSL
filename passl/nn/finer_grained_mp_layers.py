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
from paddle.distributed import fleet
from paddle.distributed.fleet.base import topology as tp
from paddle.distributed.fleet.layers.mpu.random import get_rng_state_tracker

__all__ = [
    "finer_grained_rowsharded_linear",
    "finer_grained_columnsharded_linear",
    "finer_grained_row_parallel_linear",
    "finer_grained_column_parallel_linear",
    "FinerGrainedRowParallelLinear",
    "FinerGrainedColumnParallelLinear"
]

def get_finer_grained_model_parallel_communication_info():
    hcg = fleet.get_hybrid_communicate_group()
    mp_rank = hcg.get_model_parallel_rank()
    mp_ranks = hcg.get_model_parallel_world_size()
    assert hasattr(hcg, '_mp_ring_comm_group'), "hcg must have _mp_ring_comm_group, you need to initialize model parallel ring group first"
    mp_ring_comm_group = hcg.get_model_parallel_ring_group()
    mp_group = hcg.get_model_parallel_group()

    next_mp_rank = (mp_rank + 1) % len(mp_group.ranks)
    prev_mp_rank = (mp_rank - 1 + len(mp_group.ranks)) % len(mp_group.ranks)
    send_group = mp_ring_comm_group[f'mp_{mp_rank}to{next_mp_rank}']
    recv_group = mp_ring_comm_group[f'mp_{prev_mp_rank}to{mp_rank}']
    send_dst = mp_group.ranks[next_mp_rank]
    recv_src = mp_group.ranks[prev_mp_rank]

    return mp_rank, mp_ranks, mp_group, send_group, recv_group, send_dst, recv_src

def finer_grained_rowsharded_linear(x, weight, bias=None, transpose_y=False, name=None):
    """
    y = x * weight + b = matmul(x, weight) + b
    """

    mp_rank, mp_ranks, mp_group, send_group, recv_group, send_dst, recv_src = \
        get_finer_grained_model_parallel_communication_info()

    hidden_size = x.shape[-1]
    assert hidden_size % mp_ranks == 0, f"hidden_size {hidden_size} must be divided by mp_ranks {mp_ranks}"
    micro_hidden_size = hidden_size // mp_ranks

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
        yi = paddle.matmul(xi, wi, transpose_y=transpose_y)

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

def finer_grained_rowsharded_linear_grad(dy, x, weight, bias=None, name=None):
    mp_rank, mp_ranks, mp_group, send_group, recv_group, send_dst, recv_src = \
        get_finer_grained_model_parallel_communication_info()

    hidden_size = x.shape[-1]
    assert hidden_size % mp_ranks == 0, f"hidden_size {hidden_size} must be divided by mp_ranks {mp_ranks}"
    micro_hidden_size = hidden_size // mp_ranks

    # reverse order [mp_ranks-1, ..., 1, 0]
    cal_index = list(range(mp_ranks-1, -1, -1))
    # shift
    shift = mp_ranks - mp_rank
    cal_index = cal_index[shift:] + cal_index[:shift]

    # Note(GuoxiaWang): we reshape inplace to avoid allocate memory, finally we will reshape back.
    dy_shape = dy.shape
    x_shape = x.shape

    w_shape = weight.shape
    if len(w_shape) == 2:
        with paddle.no_grad():
            dy.reshape_([-1, dy_shape[-1]])
            x.reshape_([-1, x_shape[-1]])
    else:
        with paddle.no_grad():
            dy.reshape_([-1, dy_shape[-2], dy_shape[-1]])
            x.reshape_([-1, x_shape[-2], x_shape[-1]])

    for idx, t in enumerate(cal_index):
        start = t * micro_hidden_size
        end = start + micro_hidden_size

        # slice and calculate matmul
        xi = paddle.slice(x, axes=[-1], starts=[start], ends=[end])
        if len(w_shape) == 2:
            dwi = paddle.matmul(xi, dy, transpose_x=True)
        else:
            dwi = paddle.bmm(xi.transpose((0, 2, 1)), dy)

        # we need to sync and get received
        if idx > 0:
            task_send.wait()
            task_recv.wait()
            dwi_send = dwi + dwi_recv
        else:
            dwi_send = dwi

        # launch async send and recv
        if idx < mp_ranks-1:
            if mp_rank % 2 == 0:
                task_send = dist.isend(dwi_send, dst=send_dst, group=send_group)
            else:
                dwi_recv = paddle.zeros_like(dwi_send)
                task_recv = dist.irecv(dwi_recv, src=recv_src, group=recv_group)

            if mp_rank % 2 == 0:
                dwi_recv = paddle.zeros_like(dwi_send)
                task_recv = dist.irecv(dwi_recv, src=recv_src, group=recv_group)
            else:
                task_send = dist.isend(dwi_send, dst=send_dst, group=send_group)

    with paddle.no_grad():
        dy.reshape_(dy_shape)
        x.reshape_(x_shape)
        dwi_send.reshape_(w_shape)


    if bias is not None:
        bias_grad = paddle.sum(dy, axis=list(range(len(dy.shape)-1)))
        task_bias_grad = dist.all_reduce(bias_grad, group=mp_group, sync_op=False)

    weight_grad = dwi_send
    x_grad = finer_grained_columnsharded_linear(dy, weight, transpose_y=True)

    if bias is not None:
        task_bias_grad.wait()
    else:
        bias_grad = None

    return x_grad, weight_grad, bias_grad


def finer_grained_columnsharded_linear(x, weight, bias=None, transpose_y=False, name=None):
    """
    y = x * weight + b = matmul(x, weight) + b
    """

    mp_rank, mp_ranks, mp_group, send_group, recv_group, send_dst, recv_src = \
        get_finer_grained_model_parallel_communication_info()

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
        yi = paddle.matmul(x, wi, transpose_y=transpose_y)

        y.append(yi)

        # we need to sync and get received wi
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


def finer_grained_columnsharded_linear_grad(dy, x, weight, bias=None, name=None):
    mp_rank, mp_ranks, mp_group, send_group, recv_group, send_dst, recv_src = \
        get_finer_grained_model_parallel_communication_info()

    hidden_size = dy.shape[-1]
    assert hidden_size % mp_ranks == 0, f"hidden_size {hidden_size} must be divided by mp_ranks {mp_ranks}"
    micro_hidden_size = hidden_size // mp_ranks

    # reverse order [mp_ranks-1, ..., 1, 0]
    cal_index = list(range(mp_ranks-1, -1, -1))
    # shift
    shift = mp_ranks - mp_rank
    cal_index = cal_index[shift:] + cal_index[:shift]

    # Note(GuoxiaWang): we reshape inplace to avoid allocate memory, finally we will reshape back.
    dy_shape = dy.shape
    x_shape = x.shape
    w_shape = weight.shape
    if len(w_shape) == 2:
        with paddle.no_grad():
            dy.reshape_([-1, dy_shape[-1]])
            x.reshape_([-1, x_shape[-1]])
    else:
        with paddle.no_grad():
            dy.reshape_([-1, dy_shape[-2], dy_shape[-1]])
            x.reshape_([-1, x_shape[-2], x_shape[-1]])

    for idx, t in enumerate(cal_index):
        start = t * micro_hidden_size
        end = start + micro_hidden_size

        # slice and calculate matmul
        dyi = paddle.slice(dy, axes=[-1], starts=[start], ends=[end])
        if len(w_shape) == 2:
            dwi = paddle.matmul(x, dyi, transpose_x=True)
        else:
            dwi = paddle.bmm(x.transpose((0, 2, 1)), dyi)

        # we need to sync and get received
        if idx > 0:
            task_send.wait()
            task_recv.wait()
            dwi_send = dwi + dwi_recv
        else:
            dwi_send = dwi

        # launch async send and recv
        if idx < mp_ranks-1:
            if mp_rank % 2 == 0:
                task_send = dist.isend(dwi_send, dst=send_dst, group=send_group)
            else:
                dwi_recv = paddle.zeros_like(dwi_send)
                task_recv = dist.irecv(dwi_recv, src=recv_src, group=recv_group)

            if mp_rank % 2 == 0:
                dwi_recv = paddle.zeros_like(dwi_send)
                task_recv = dist.irecv(dwi_recv, src=recv_src, group=recv_group)
            else:
                task_send = dist.isend(dwi_send, dst=send_dst, group=send_group)

    with paddle.no_grad():
        dy.reshape_(dy_shape)
        x.reshape_(x_shape)
        dwi_send.reshape_(w_shape)

    if bias is not None:
        bias_grad = paddle.sum(dy, axis=list(range(len(dy.shape)-1)))
        task_bias_grad = dist.all_reduce(bias_grad, group=mp_group, sync_op=False)

    weight_grad = dwi_send
    x_grad = finer_grained_rowsharded_linear(dy, weight, transpose_y=True)
    if bias is not None:
        task_bias_grad.wait()
    else:
        bias_grad = None

    return x_grad, weight_grad, bias_grad


class FinerGrainedRowShardedLinearFunction(PyLayer):
    @staticmethod
    def forward(ctx, x, weight, bias=None, split_x=False, split_axis=0, gather_y=False, name=None):
        # Note(GuoxiaWang): save input dtype to recover grad dtype for amp
        ctx.x_dtype = x.dtype
        ctx.weight_dtype = weight.dtype
        ctx.name = name
        ctx.has_bias = bias is not None
        if bias is not None:
            ctx.bias_dtype = bias.dtype

        if split_x:
            world_size = tp._HYBRID_PARALLEL_GROUP.get_model_parallel_world_size()
            mp_rank = tp._HYBRID_PARALLEL_GROUP.get_model_parallel_world_rank()
            x = paddle.split(x, world_size, axis=split_axis)[mp_rank]

        if ctx.has_bias:
            ctx.save_for_backward(x, weight, bias)
        else:
            ctx.save_for_backward(x, weight)
        ctx.split_x = split_x
        ctx.gather_y = gather_y
        ctx.split_axis = split_axis

        # Note(GuoxiaWang): it will auto cast dtype when enabling amp
        y = finer_grained_rowsharded_linear(x, weight, bias)

        if gather_y:
            mp_group = tp._HYBRID_PARALLEL_GROUP.get_model_parallel_group()
            tensor_list = []
            dist.all_gather(tensor_list, y, group=mp_group)
            y = paddle.concat(tensor_list, axis=0)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.has_bias:
            x, weight, bias = ctx.saved_tensor()
        else:
            x, weight = ctx.saved_tensor()
            bias = None
        if ctx.gather_y:
            world_size = tp._HYBRID_PARALLEL_GROUP.get_model_parallel_world_size()
            mp_rank = tp._HYBRID_PARALLEL_GROUP.get_model_parallel_world_rank()
            grad_output = paddle.split(grad_output, world_size, axis=0)[mp_rank]

        # Note(GuoxiaWang): it needs to be manually converted to the type when enabling the amp.
        if x.dtype != grad_output.dtype:
            x = x.astype(grad_output.dtype)
        if weight.dtype != grad_output.dtype:
            weight = weight.astype(grad_output.dtype)
        if bias is not None and bias.dtype != grad_output.dtype:
            bias = bias.astype(grad_output.dtype)

        x_grad, weight_grad, bias_grad = finer_grained_rowsharded_linear_grad(grad_output, x, weight, bias)

        if ctx.split_x:
            mp_group = tp._HYBRID_PARALLEL_GROUP.get_model_parallel_group()
            tensor_list = []
            dist.all_gather(tensor_list, x_grad, group=mp_group)
            x_grad = paddle.concat(tensor_list, axis=ctx.split_axis)

        # Note(GuoxiaWang): recover the grad dtype
        if x_grad.dtype != ctx.x_dtype:
            x_grad = x_grad.astype(ctx.x_dtype)
        if weight_grad.dtype != ctx.weight_dtype:
            weight_grad = weight_grad.astype(ctx.weight_dtype)
        if bias is not None and bias_grad.dtype != ctx.bias_dtype:
            bias_grad = bias_grad.astype(ctx.bias_dtype)

        if bias is not None:
            return x_grad, weight_grad, bias_grad
        else:
            return x_grad, weight_grad


class FinerGrainedColumnShardedLinearFunction(PyLayer):
    @staticmethod
    def forward(ctx, x, weight, bias=None, split_x=False, split_axis=0, gather_y=False, name=None):
        # Note(GuoxiaWang): save input dtype to recover grad dtype for amp
        ctx.x_dtype = x.dtype
        ctx.weight_dtype = weight.dtype
        ctx.name = name
        ctx.has_bias = bias is not None
        if bias is not None:
            ctx.bias_dtype = bias.dtype

        if split_x:
            world_size = tp._HYBRID_PARALLEL_GROUP.get_model_parallel_world_size()
            mp_rank = tp._HYBRID_PARALLEL_GROUP.get_model_parallel_world_rank()
            x = paddle.split(x, world_size, axis=split_axis)[mp_rank]

        if ctx.has_bias:
            ctx.save_for_backward(x, weight, bias)
        else:
            ctx.save_for_backward(x, weight)
        ctx.split_x = split_x
        ctx.gather_y = gather_y
        ctx.split_axis = split_axis

        # Note(GuoxiaWang): it will auto cast dtype when enable amp
        y = finer_grained_columnsharded_linear(x, weight, bias)

        if gather_y:
            mp_group = tp._HYBRID_PARALLEL_GROUP.get_model_parallel_group()
            tensor_list = []
            dist.all_gather(tensor_list, y, group=mp_group)
            y = paddle.concat(tensor_list, axis=0)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.has_bias:
            x, weight, bias = ctx.saved_tensor()
        else:
            x, weight = ctx.saved_tensor()
            bias = None
        if ctx.gather_y:
            world_size = tp._HYBRID_PARALLEL_GROUP.get_model_parallel_world_size()
            mp_rank = tp._HYBRID_PARALLEL_GROUP.get_model_parallel_world_rank()
            grad_output = paddle.split(grad_output, world_size, axis=0)[mp_rank]

        # Note(GuoxiaWang): it needs to be manually converted to the type when enabling the amp.
        if x.dtype != grad_output.dtype:
            x = x.astype(grad_output.dtype)
        if weight.dtype != grad_output.dtype:
            weight = weight.astype(grad_output.dtype)
        if bias is not None and bias.dtype != grad_output.dtype:
            bias = bias.astype(grad_output.dtype)

        x_grad, weight_grad, bias_grad = finer_grained_columnsharded_linear_grad(grad_output, x, weight, bias)

        if ctx.split_x:
            mp_group = tp._HYBRID_PARALLEL_GROUP.get_model_parallel_group()
            tensor_list = []
            dist.all_gather(tensor_list, x_grad, group=mp_group)
            x_grad = paddle.concat(tensor_list, axis=ctx.split_axis)

        # Note(GuoxiaWang): recover the grad dtype
        if x_grad.dtype != ctx.x_dtype:
            x_grad = x_grad.astype(ctx.x_dtype)
        if weight_grad.dtype != ctx.weight_dtype:
            weight_grad = weight_grad.astype(ctx.weight_dtype)
        if bias is not None and bias_grad.dtype != ctx.bias_dtype:
            bias_grad = bias_grad.astype(ctx.bias_dtype)

        if bias is not None:
            return x_grad, weight_grad, bias_grad
        else:
            return x_grad, weight_grad


def finer_grained_row_parallel_linear(x, weight, bias=None, split_x=False, split_axis=0, gather_y=False, name=None):
    return FinerGrainedRowShardedLinearFunction.apply(x, weight, bias=bias, split_x=split_x, split_axis=split_axis, gather_y=gather_y, name=name)


def finer_grained_column_parallel_linear(x, weight, bias=None, split_x=False, split_axis=0, gather_y=False, name=None):
    return FinerGrainedColumnShardedLinearFunction.apply(x, weight, bias=bias, split_x=split_x, split_axis=split_axis, gather_y=gather_y, name=name)


class FinerGrainedRowParallelLinear(paddle.nn.Layer):
    def __init__(
        self,
        in_features,
        out_features,
        weight_attr=None,
        bias_attr=None,
        input_split=False,
        input_split_axis=0,
        gather_output=False,
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
        self.input_split = input_split
        self.input_split_axis = input_split_axis
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
        if self.bias is not None:
            self.bias.is_distributed = self.weight.is_distributed

        self.linear = finer_grained_row_parallel_linear


    def forward(self, x):
        output = self.linear(
            x,
            self.weight,
            self.bias,
            split_x=self.input_split and self.is_mp,
            split_axis=self.input_split_axis,
            gather_y=self.gather_output and self.is_mp,
            name=self._name
        )
        return output


class FinerGrainedColumnParallelLinear(paddle.nn.Layer):
    def __init__(
        self,
        in_features,
        out_features,
        weight_attr=None,
        bias_attr=None,
        input_split=False,
        input_split_axis=0,
        gather_output=False,
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
        self.input_split = input_split
        self.input_split_axis = input_split_axis
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
        if self.bias is not None:
            self.bias.is_distributed = self.weight.is_distributed

        self.linear = finer_grained_column_parallel_linear

    def forward(self, x):
        output = self.linear(
            x,
            self.weight,
            self.bias,
            split_x=self.input_split and self.is_mp,
            split_axis=self.input_split_axis,
            gather_y=self.gather_output and self.is_mp,
            name=self._name
        )
        return output
