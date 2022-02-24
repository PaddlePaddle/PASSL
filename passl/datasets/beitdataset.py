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

import paddle
from .folder import DatasetFolder

from .preprocess import build_transforms, MaskingGenerator
from .builder import DATASETS
from ..utils.misc import accuracy


@DATASETS.register()
class BEiT_ImageNet(DatasetFolder):
    cls_filter = None

    def __init__(self,
                 dataroot,
                 common_transforms=None,
                 patch_transforms=None,
                 visual_token_transforms=None,
                 masking_generator=None):
        super(BEiT_ImageNet, self).__init__(dataroot,
                                            cls_filter=self.cls_filter)

        self.common_transform = build_transforms(common_transforms)
        self.patch_transform = build_transforms(patch_transforms)
        self.visual_token_transform = build_transforms(visual_token_transforms)
        self.masked_position_generator = MaskingGenerator(**masking_generator)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        # Only Used For Debug The DataAug Module.
        #path = 'data/ILSVRC2012/train/n13040303/n13040303_1206.jpeg'
        #target = 14
        sample = self.loader(path)
        for_patches, for_visual_tokens = self.common_transform(sample)
        return \
            self.patch_transform(for_patches), \
            self.visual_token_transform(for_visual_tokens), \
            self.masked_position_generator()

    def evaluate(self, preds, labels, topk=(1, 5)):

        eval_res = {}
        eval_res['acc1'], eval_res['acc5'] = accuracy(preds, labels, topk)

        return eval_res
