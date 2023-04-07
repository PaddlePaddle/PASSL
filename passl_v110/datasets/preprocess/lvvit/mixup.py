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

""" Mixup and Cutmix

Papers:
mixup: Beyond Empirical Risk Minimization (https://arxiv.org/abs/1710.09412)

CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features (https://arxiv.org/abs/1905.04899)

Code Reference:
CutMix: https://github.com/clovaai/CutMix-PyTorch

Hacked together by / Copyright 2020 Ross Wightman
"""

import paddle
import paddle.nn.functional as F
from paddle.vision.ops import roi_align


def astype(x, dtype):
    return x if x.dtype == dtype else x.astype(dtype)


def one_hot(x, num_classes, on_value=1., off_value=0.):
    return F.one_hot(astype(x, paddle.int64), num_classes) * (
        on_value - off_value) + off_value


def get_featuremaps(label_maps_topk, num_classes):
    B, _, topk, H, W = label_maps_topk.shape
    src, index = label_maps_topk[:, 0].flatten(), label_maps_topk[:, 1].flatten(
    )
    one_hot = F.one_hot(astype(index, paddle.int64), num_classes)
    label_maps = one_hot * astype(src, paddle.float32).unsqueeze(-1)
    label_maps = label_maps.reshape([B, topk, H, W, num_classes]).sum(1)
    label_maps = label_maps.transpose([0, 3, 1, 2])
    return label_maps


def get_label(label_maps, batch_coords, label_size=1):
    '''
    Adapted from https://github.com/naver-ai/relabel_imagenet/blob/main/utils/relabel_functions.py
    Here we generate label for patch tokens and cls token separately and concat them together if given label_size>1
    '''
    num_batches = label_maps.shape[0]
    boxes = astype(batch_coords, paddle.float32) * label_maps.shape[3]
    num_boxes = paddle.ones([num_batches], dtype=paddle.int32)
    target_label = roi_align(label_maps, boxes, num_boxes,
                             (label_size, label_size))
    if label_size > 1:
        target_label_cls = roi_align(label_maps, boxes, num_boxes, (1, 1))
        B, C, H, W = target_label.shape
        target_label = target_label.reshape([B, C, H * W])
        target_label = paddle.concat(
            [target_label_cls.reshape([B, C, 1]), target_label], axis=2)
    target_label = F.softmax(target_label.squeeze(), axis=1)
    return target_label


def get_labelmaps_with_coords(label_maps_topk,
                              num_classes,
                              on_value=1.,
                              off_value=0.,
                              label_size=1):
    '''
    Adapted from https://github.com/naver-ai/relabel_imagenet/blob/main/utils/relabel_functions.py
    Generate the target label map for training from the given bbox and raw label map
    '''
    # trick to get coords_map from label_map
    random_crop_coords = label_maps_topk[:, 2, 0, 0, :4].reshape([-1, 4])
    random_crop_coords[:, 2:] += random_crop_coords[:, :2]

    # trick to get ground truth from label_map
    ground_truth = astype(label_maps_topk[:, 2, 0, 0, 5],
                          paddle.int64).flatten()
    ground_truth = one_hot(
        ground_truth, num_classes, on_value=on_value, off_value=off_value)

    # get full label maps from raw topk labels
    label_maps = get_featuremaps(
        label_maps_topk=label_maps_topk, num_classes=num_classes)

    # get token-level label and ground truth
    token_label = get_label(
        label_maps=label_maps,
        batch_coords=random_crop_coords,
        label_size=label_size)
    B, C = token_label.shape[:2]
    token_label = token_label * on_value + off_value
    if label_size == 1:
        return paddle.concat(
            [ground_truth.reshape([B, C, 1]), token_label.reshape([B, C, 1])],
            axis=2)
    else:
        return paddle.concat(
            [ground_truth.reshape([B, C, 1]), token_label], axis=2)


def mixup_target(target, num_classes, lam=1., smoothing=0.0, label_size=1):
    '''
    generate and mix target from the given label maps
    target: label maps/ label maps with coords
    num_classes: number of classes for the target
    lam: lambda for mixup target
    '''
    off_value = smoothing / num_classes
    on_value = 1. - smoothing + off_value
    if target.ndim > 2:
        if target.shape[1] == 3:
            y1 = get_labelmaps_with_coords(
                target,
                num_classes,
                on_value=on_value,
                off_value=off_value,
                label_size=label_size)
            y2 = y1.flip(0)
        else:
            raise ValueError("Not supported label type")
    else:
        y1 = one_hot(
            target, num_classes, on_value=on_value, off_value=off_value)
        y2 = one_hot(
            target.flip(0), num_classes, on_value=on_value, off_value=off_value)

    return y1 * lam + y2 * (1. - lam)


class Mixup:
    """
    Args:
        lam: lambda for mixup target
        smoothing (float): apply label smoothing to the mixed target tensor
        num_classes (int): number of classes for target
    """

    def __init__(self, lam=1., smoothing=0.0, label_size=1, num_classes=1000):
        self.lam = lam
        self.smoothing = smoothing
        self.label_size = label_size
        self.num_classes = num_classes

    def __call__(self, x, target):
        target = mixup_target(
            target,
            self.num_classes,
            lam=self.lam,
            smoothing=self.smoothing,
            label_size=self.label_size)
        if len(target.shape) == 1:
            target = mixup_target(
                target, self.num_classes, smoothing=self.smoothing)

        return x, target
