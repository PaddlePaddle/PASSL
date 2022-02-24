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
import numpy as np
import math
import paddle
from paddle.io import DistributedBatchSampler

from ..utils.registry import Registry, build_from_config
from .preprocess.builder import build_transforms
from .preprocess.mixup import Mixup

DATASETS = Registry("DATASET")


class DistributedRepeatedAugSampler(DistributedBatchSampler):
    """
    based on https://github.com/facebookresearch/deit/blob/main/samplers.py
    """
    def __init__(self,
                 dataset,
                 batch_size,
                 num_replicas=None,
                 rank=None,
                 shuffle=False,
                 drop_last=False):
        super(DistributedRepeatedAugSampler,
              self).__init__(dataset, batch_size, num_replicas, rank, shuffle,
                             drop_last)
        self.num_samples = int(math.ceil(len(self.dataset) * 3.0 / self.nranks))
        self.total_size = self.num_samples * self.nranks
        self.num_selected_samples = int(
            math.floor(len(self.dataset) // 256 * 256 / self.nranks))

    def __iter__(self):
        num_samples = len(self.dataset)
        indices = np.arange(num_samples).tolist()
        if self.shuffle:
            np.random.RandomState(self.epoch).shuffle(indices)
            self.epoch += 1

        indices = [ele for ele in indices for i in range(3)]
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.local_rank:self.total_size:self.nranks]
        assert len(indices) == self.num_samples
        _sample_iter = iter(indices[:self.num_selected_samples])

        batch_indices = []
        for idx in _sample_iter:
            batch_indices.append(idx)
            if len(batch_indices) == self.batch_size:
                yield batch_indices
                batch_indices = []
        if not self.drop_last and len(batch_indices) > 0:
            yield batch_indices

    def __len__(self):
        num_samples = self.num_selected_samples
        num_samples += int(not self.drop_last) * (self.batch_size - 1)
        return num_samples // self.batch_size


def build_dataset(cfg):
    return build_from_config(cfg, DATASETS)


def build_dataloader(cfg, device):
    cfg_ = copy.deepcopy(cfg)
    loader_cfg = cfg_.pop('loader')
    dataset_cfg = cfg_.pop('dataset')
    sampler_cfg = cfg_.pop('sampler')

    mixup_cfg = dataset_cfg.pop(
        'batch_transforms') if 'batch_transforms' in dataset_cfg else None

    dataset = build_dataset(dataset_cfg)

    sampler_name = sampler_cfg.pop('name', 'DistributedBatchSampler')

    sampler = eval("{}".format(sampler_name))(dataset, **sampler_cfg)

    dataloader = paddle.io.DataLoader(dataset,
                                      batch_sampler=sampler,
                                      place=device,
                                      **loader_cfg)

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
