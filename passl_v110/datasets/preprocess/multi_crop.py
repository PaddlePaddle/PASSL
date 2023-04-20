# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
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
# Taken and modified for PASSL from timm:
#   https://github.com/yang-ruixin/PyTorch-Image-Models-Multi-Label-Classification/blob/main/timm/data/auto_augment.py
# AutoAugment and RandAug Implementation adapted from:
#    https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/autoaugment.py
# AugMix adapted from:
#    https://github.com/google-research/augmix
# Copyright 2020 Ross Wightman

from ..preprocess import build_transforms

class MultiCrop():
    def __init__(self,
                 global_transform1,
                 global_transform2,
                 local_transform,
                 local_crops_number):
        # first global crop
        self.global_transform1 = build_transforms(global_transform1)
        # second global crop
        self.global_transform2 = build_transforms(global_transform2)
        # transformation for the local small crops
        self.local_crops_number = local_crops_number
        self.local_transform = build_transforms(local_transform)

    def __call__(self, image):
        crops = []
        crops.append(self.global_transform1(image))
        crops.append(self.global_transform2(image))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transform(image))
        return crops

from paddle.vision.transforms import (
    Compose,
    Transpose,
    ColorJitter,
    RandomResizedCrop,
    RandomHorizontalFlip,
)

from ..preprocess.transforms import (
    RandomApply,
    Solarization,
    GaussianBlur,
    NormalizeImage,
    RandomGrayscale,
)


class MultiCropv2():
    def __init__(self,
                 global_crops_scale,
                 local_crops_scale,
                 local_crops_number):
        flip_and_color_jitter = Compose([
            RandomHorizontalFlip(prob=0.5),
            RandomApply(
                [ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            RandomGrayscale(p=0.2)
        ])

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        normalize = Compose([
            Transpose(),
            NormalizeImage(scale='1.0/255.0', mean=mean, std=std),
        ])

        # first global crop
        self.global_transform1 = Compose([
            RandomResizedCrop(224, scale=(0.4, 1.), interpolation='bicubic'),
            flip_and_color_jitter,
            RandomApply(
                [GaussianBlur(_PIL=True)],
                p=1.0
            ),
            normalize,
        ])

        # second global crop
        self.global_transform2 = Compose([
            RandomResizedCrop(224, scale=(0.4, 1.), interpolation='bicubic'),
            flip_and_color_jitter,
            RandomApply(
                [GaussianBlur(_PIL=True)],
                p=0.1
            ),
            RandomApply(
                [Solarization()],
                p=0.2
            ),
            normalize,
        ])
        # transformation for the local small crops
        self.local_crops_number = local_crops_number
        self.local_transform = Compose([
            RandomResizedCrop(96, scale=(0.05, 0.4), interpolation='bicubic'),
            flip_and_color_jitter,
            RandomApply(
                [GaussianBlur(_PIL=True)],
                p=0.5
            ),
            normalize
        ])

    def __call__(self, image):
        crops = []
        crops.append(self.global_transform1(image))
        crops.append(self.global_transform2(image))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transform(image))
        return crops

from .builder import TRANSFORMS
TRANSFORMS.register(MultiCrop)
TRANSFORMS.register(MultiCropv2)
