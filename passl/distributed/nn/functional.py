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
import paddle.distributed as dist
from paddle.autograd import PyLayer


def ensure_divisibility(numerator, denominator):
    """Ensure that numerator is divisible by the denominator."""
    assert numerator % denominator == 0, '{} is not divisible by {}'.format(
        numerator, denominator)


def split(tensor, axis=0, group=None):
    return _Split.apply(tensor, axis, group)


def all_gather(tensor, group=None):
    """
    Gathers tensors from the whole group in a list.

    Arguments:
        tensor (Tensor): Tensor to be broadcast from current process.
        group (ProcessGroup, optional): The process group to work on.

    Returns:
        tuple([Tensor]): Output of the collective.

    """
    return _AllGather.apply(tensor, group)


def softmax(tensor, axis=-1, group=None):
    return ParallelSoftmax.apply(tensor, axis, group)


def reshard_transpose(input, in_axis, out_axis, group):
    """ N, S, R, C => N, R, S, C using sync all_to_all and reshard.
        For example:
            in_axis = 1
            out_axis = 2
            nranks = 8
            input shape = [N, S, R, C]

            output shape = [N, S/8, 8*R, C]
    """

    nranks = 1 if group is None else group.nranks
    if nranks == 1:
        return input

    ensure_divisibility(input.shape[in_axis], nranks)
    input = paddle.concat(
        paddle.split(
            input, nranks, axis=in_axis), axis=0)

    if not input.stop_gradient:
        output = All2All.apply(input, in_axis=in_axis, out_axis=out_axis, group=group)
    else:
        output = _all_to_all(input, in_axis=in_axis, out_axis=out_axis, group=group)

    output = paddle.concat(
        paddle.split(
            output, nranks, axis=0), axis=out_axis)
    return output


class _Split(PyLayer):
    @staticmethod
    def forward(ctx, tensor, axis, group):
        ctx.group = group
        ctx.axis = axis

        rank = dist.get_rank()
        src_rank_in_group = group.get_group_rank(rank)
        out = paddle.split(tensor, group.nranks, axis=axis)[src_rank_in_group]

        return out

    @staticmethod
    def backward(ctx, grad_output):
        tensor_list = []
        dist.all_gather(tensor_list, grad_output, group=ctx.group)
        grad = paddle.concat(tensor_list, axis=ctx.axis)
        return grad


class _Reduce_Scatter(PyLayer):
    @staticmethod
    def forward(ctx, tensor, group, *input_tensor_list):
        ctx.group = group
        dist.reduce_scatter(tensor, list(input_tensor_list), group=group)
        return tensor

    @staticmethod
    def backward(ctx, grad_output):
        return (None) + _AllGather.apply(grad_output, ctx.group)


class _AllGather(PyLayer):
    @staticmethod
    def forward(ctx, tensor, group):
        ctx.group = group

        out_tensor_list = []
        dist.all_gather(out_tensor_list, tensor, group=group)
        return tuple(out_tensor_list)

    @staticmethod
    def backward(ctx, *grad_outputs):
        rank = dist.get_rank()
        src_rank_in_group = ctx.group.get_group_rank(rank)
        gx = paddle.empty_like(grad_outputs[src_rank_in_group])
        _Reduce_Scatter.apply(gx, ctx.group, *grad_outputs)
        return gx

class ParallelSoftmax(PyLayer):
    @staticmethod
    def forward(ctx, tensor, axis, group):
        ctx.group = group
        ctx.axis = axis
        ctx.dtype = tensor.dtype
        assert axis == -1 or len(tensor.shape) - 1, \
            f"Only support lastest axis for ParallelSoftmax, but got {axis}."

        nranks = 1 if group is None else group.nranks
        ctx.nranks = nranks

        max_value = paddle.max(tensor, axis=axis, keepdim=True)
        # local to global
        if nranks > 1:
            dist.all_reduce(max_value, dist.ReduceOp.MAX, group=group)
        tensor = paddle.exp(tensor - max_value)
        sum_exp = paddle.sum(tensor, axis=axis, keepdim=True)
        # local to global
        if nranks > 1:
            dist.all_reduce(sum_exp, dist.ReduceOp.SUM, group=group)
        out = tensor / sum_exp

        ctx.save_for_backward(out)

        return out

    @staticmethod
    def backward(ctx, grad_output):
        [out] = ctx.saved_tensor()

        grad_sum = paddle.sum(grad_output * out, axis=ctx.axis, keepdim=True)
        # local to global
        if ctx.nranks > 1:
            dist.all_reduce(grad_sum, dist.ReduceOp.SUM, group=ctx.group)
        grad = out * (grad_output - grad_sum)
        if grad.dtype != ctx.dtype:
            grad = grad.astype(ctx.dtype)
        return grad


@paddle.no_grad()
def _all_to_all(tensor, in_axis=-1, out_axis=-1, sync_op=True, group=None):
    tensor_shape = list(tensor.shape)

    out = paddle.zeros(tensor_shape, tensor.dtype)
    out.stop_gradient = tensor.stop_gradient
    task = group.process_group.alltoall(tensor, out)

    if sync_op:
        task.wait()
        return out
    else:
        return task, out


class All2All(PyLayer):
    @staticmethod
    def forward(ctx, input, in_axis=-1, out_axis=-1, group=None):
        ctx.in_axis = in_axis
        ctx.out_axis = out_axis
        ctx.group = group
        return _all_to_all(input, in_axis=in_axis, out_axis=out_axis, group=group)

    @staticmethod
    def backward(ctx, grad_output):
        return _all_to_all(
            grad_output, in_axis=ctx.out_axis, out_axis=ctx.in_axis, group=ctx.group)
