# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserve.
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

import os
import numpy as np
import paddle

from .folder import DatasetFolder
from .preprocess import build_transforms
from .builder import DATASETS
from ..utils.misc import accuracy


@DATASETS.register()
class LVViT_ImageNet(DatasetFolder):
    def __init__(self, dataroot, labelroot, transforms=None):
        super().__init__(dataroot)

        self.labelroot = labelroot
        self.transforms = transforms
        if transforms is not None:
            self.transform = build_transforms(transforms)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)

        score_path = os.path.join(
            self.labelroot,
            '/'.join(path.split('/')[-2:]).split('.')[0] + '.npy')
        score_maps = np.load(score_path).astype(np.float32)

        if self.transform is not None:
            sample, score_maps = self.transform((sample, score_maps))

        # append ground truth after coords
        score_maps[-1, 0, 0, 5] = target
        target = paddle.to_tensor(score_maps)

        return sample, target

    def evaluate(self, preds, labels, topk=(1, 5)):

        eval_res = {}
        eval_res['acc1'], eval_res['acc5'] = accuracy(preds, labels, topk)

        return eval_res
