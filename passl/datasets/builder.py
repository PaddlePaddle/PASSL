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

import copy
import paddle

from ..utils.registry import Registry, build_from_config
from .preprocess.builder import build_transforms
from .preprocess.mixup import Mixup

DATASETS = Registry("DATASET")


def build_dataset(cfg):
    return build_from_config(cfg, DATASETS)


def build_dataloader(cfg):
    cfg_ = copy.deepcopy(cfg)
    dataset_cfg = cfg_.pop('dataset')
    sampler_cfg = cfg_.pop('sampler')

    mixup_cfg = dataset_cfg.pop(
        'batch_transforms') if 'batch_transforms' in dataset_cfg else None

    dataset = build_dataset(dataset_cfg)

    sampler = paddle.io.DistributedBatchSampler(dataset, **sampler_cfg)

    dataloader = paddle.io.DataLoader(dataset, batch_sampler=sampler, **cfg_)

    #setup mixup / cutmix
    mixup_fn = None
    if mixup_cfg is not None:
        mixup_cfg = mixup_cfg[0]
        mixup_active = mixup_cfg['mixup_alpha'] > 0 or mixup_cfg[
            'cutmix_alpha'] > 0. or mixup_cfg['cutmix_minmax'] != ''  # noqa
        if mixup_active:
            mixup_fn = Mixup(mixup_alpha=mixup_cfg['mixup_alpha'],
                             cutmix_alpha=mixup_cfg['cutmix_alpha'],
                             prob=mixup_cfg['prob'],
                             switch_prob=mixup_cfg['switch_prob'],
                             mode=mixup_cfg['mode'])

    return dataloader, mixup_fn
