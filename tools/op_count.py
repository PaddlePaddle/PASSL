# -----------------------------------------------------------------------------------------
# Here it is mainly used to calculate the FLOPs of the operator.
#
# FLOPs is abbreviation of floating operations which includes mul / add / div ... etc.
# The calculation of FLOPs is not fixed，The following papers all propose FLOPs calculation.
#
# - paper: Practical Guidelines for Efficient CNN Architecture Design
#          arxiv 1807.11164 from Megvii
# - paper: Pruning Convolutional Neural Networks for Resource Efficient Inference.
#          arxiv 1611.06440 from NVIDIA
#
# In fact, the addition operation is indeterminate in the operation of matrices. To facilitate
# the comparison of FLOPs, we mainly consider the multiplication operation.
# Of course you can also define your own FLOPs for the calculation of operators.
#
# Currently supported op: - nn.Conv2D
#                         - nn.Linear
#                         - nn.BatchNrom2D
#                         - nn.AvgPool
#                         - nn.GELU
#                         - nn.LayerNorm
#                         - coming soon ...
#
# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License")
# --------------------------------------------------------------------------------------------

import paddle
import paddle.nn as nn


def count_conv2d(m, x, y):
    x = x[0]
    kernel_ops = paddle.zeros(m.weight.shape[2:]).numel()
    bias_ops = 1 if m.bias is not None else 0
    total_ops = y.numel() * (x.shape[1] / m._groups * kernel_ops + bias_ops)
    m.flops += int(total_ops)


def count_linear(m, x, y):
    total_mul = m.weight.shape[0]
    num_elements = y.numel()
    total_ops = total_mul * num_elements
    m.flops += int(total_ops)


def count_bn(m, x, y):
    x = x[0]
    nelements = x.numel()
    if not m.training:
        total_ops = 2 * nelements
    m.flops += int(total_ops)


def count_avgpool(m, x, y):
    kernel_ops = 1
    num_elements = y.numel()
    total_ops = kernel_ops * num_elements

    m.flops += int(total_ops)


def count_gelu(m, x, y):
    gelu_ops = 8
    x = x[0]
    num_elements = x.numel()
    total_ops = gelu_ops * num_elements
    m.flops += int(total_ops)


def count_ln(m, x, y):
    x = x[0]
    num_elements = x.numel()
    if not m.training:
        total_ops = 2 * num_elements
    m.flops += int(total_ops)


def count_parameters(m, x, y):
    total_params = 0
    for p in m.parameters():
        total_params += p.numel()
    m.params[0] = int(total_params)


def count_zero_ops(m, x, y):
    m.flops += int(0)


def count_io_info(m, x, y):
    m.register_buffer('input_shape', paddle.to_tensor(x[0].shape))
    m.register_buffer('output_shape', paddle.to_tensor(y.shape))


register_hooks = {
    nn.Conv2D: count_conv2d,
    nn.BatchNorm2D: count_bn,
    nn.ReLU: count_zero_ops,
    nn.ReLU6: count_zero_ops,
    nn.Linear: count_linear,
    nn.Dropout: count_zero_ops,
    nn.AvgPool1D: count_avgpool,
    nn.AvgPool2D: count_avgpool,
    nn.GELU: count_gelu,
    nn.LayerNorm: count_ln,
}
