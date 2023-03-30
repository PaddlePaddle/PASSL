# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function

import numpy as np

from paddle.io import Dataset
from passl.utils import logger


class CommonDataset(Dataset):
    def __init__(self,
                 image_root,
                 cls_label_path,
                 transform_ops=None,
                 delimiter=" ",
                 multi_label=False,
                 class_num=None):
        if multi_label:
            assert class_num is not None, "Must set class_num when multi_label=True"
        self.multi_label = multi_label
        self.classes_num = class_num

        self._img_root = image_root
        self._cls_path = cls_label_path
        self.delimiter = delimiter
        if transform_ops:
            self._transform_ops = transform_ops

        self.images = []
        self.labels = []
        self._load_anno()

    def _load_anno(self):
        pass

    def __getitem__(self, idx):
        with open(self.images[idx], 'rb') as f:
            img = f.read()
        if self._transform_ops:
            img = self._transform_ops(img)
        if self.multi_label:
            one_hot = np.zeros([self.classes_num], dtype=np.float32)
            cls_idx = [int(e) for e in self.labels[idx].split(',')]
            for idx in cls_idx:
                one_hot[idx] = 1.0
            return (img, one_hot)
        else:
            return (img, np.int32(self.labels[idx]))

    def __len__(self):
        return len(self.images)

    @property
    def class_num(self):
        if self.multi_label:
            return self.classes_num
        return len(set(self.labels))
