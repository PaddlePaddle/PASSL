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
import paddle.vision.datasets as datasets

from .preprocess import build_transforms
from .builder import DATASETS
from ..utils.misc import accuracy


@DATASETS.register()
class ImageNet(datasets.DatasetFolder):
    def __init__(self,
                 dataroot,
                 return_label,
                 return_two_sample=False,
                 transforms=None):
        super(ImageNet, self).__init__(dataroot)

        self.return_label = return_label
        self.return_two_sample = return_two_sample
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

        if self.return_two_sample:
            sample1 = self.transform(sample)
            sample2 = self.transform(sample)
            return sample1, sample2

        if self.transform is not None:
            sample = self.transform(sample)

        if self.return_label:
            return sample, target

        return sample

    def evaluate(self, preds, labels, topk=(1, 5)):

        eval_res = {}
        eval_res['acc1'], eval_res['acc5'] = accuracy(preds, labels, topk)

        return eval_res
