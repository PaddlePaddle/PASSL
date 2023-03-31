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

import paddle

from .mixup import Mixup
from .lvvit import LVViTMixup
from ...utils.registry import Registry, build_from_config

TRANSFORMS = Registry("TRANSFORM")


def build_transform(cfg):
    return build_from_config(cfg, TRANSFORMS)


def build_transforms(cfg_list):
    transforms = []

    for cfg in cfg_list:
        transforms.append(build_transform(cfg))

    return paddle.vision.transforms.Compose(transforms)


def build_mixup(cfg):
    mixup_fn = None
    if cfg is not None:
        cfg = cfg[0]
        if cfg['name'] == 'LVViTMixup':
            mixup_fn = LVViTMixup(
                lam=cfg['lam'],
                smoothing=cfg['smoothing'],
                label_size=cfg['label_size'],
                num_classes=cfg['num_classes'])
        else:
            mixup_active = cfg['mixup_alpha'] > 0 or cfg[
                'cutmix_alpha'] > 0. or cfg['cutmix_minmax'] != ''  # noqa
            if mixup_active:
                mixup_fn = Mixup(
                    mixup_alpha=cfg['mixup_alpha'],
                    cutmix_alpha=cfg['cutmix_alpha'],
                    prob=cfg['prob'],
                    switch_prob=cfg['switch_prob'],
                    mode=cfg['mode'])
    return mixup_fn
