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
import numpy as np
from paddle.vision.datasets import Cifar10

from .preprocess import build_transforms
from .builder import DATASETS
from ..utils.misc import accuracy
from PIL import Image


@DATASETS.register()
class CIFAR10(Cifar10):
    def __init__(self,
                 datafile=None,
                 mode='train',
                 return_label=False,
                 return_two_sample=True,
                 transforms=None,
                 download=True):
        transform = build_transforms(transforms) if transforms is not None else None
        super(CIFAR10, self).__init__(datafile, mode=mode, transform=transform, download=download)

        self.return_label = return_label
        self.return_two_sample = return_two_sample

        
    def __getitem__(self, idx):
        image, label = self.data[idx]
        image = np.reshape(image, [3, 32, 32])
        image = image.transpose([1, 2, 0])

        if self.backend == 'pil':
            image = Image.fromarray(image.astype('uint8'))

        if self.return_two_sample:
            image1 = self.transform(image)
            image2 = self.transform(image)
            return image1, image2

        if self.transform is not None:
            image = self.transform(image)
        
        if self.return_label:
            return image, np.array(label).astype('int64')

        return image

    def evaluate(self, preds, labels, topk=(1, 5)):

        eval_res = {}
        eval_res['acc1'], eval_res['acc5'] = accuracy(preds, labels, topk)

        return eval_res


@DATASETS.register()
class CIFAR100(CIFAR10):
    def __init__(self,
                 datafile=None,
                 mode='train',
                 return_label=False,
                 return_two_sample=True,
                 transforms=None,
                 download=True):
        super(CIFAR100, self).__init__(datafile, mode, return_label, return_two_sample, transforms, download)

    def _init_url_md5_flag(self):
        URL_PREFIX = 'https://dataset.bj.bcebos.com/cifar/'
        CIFAR100_URL = URL_PREFIX + 'cifar-100-python.tar.gz'
        CIFAR100_MD5 = 'eb9058c3a382ffc7106e4002c42a8d85'
        MODE_FLAG_MAP = {
            'train10': 'data_batch',
            'test10': 'test_batch',
            'train100': 'train',
            'test100': 'test'
        }
        self.data_url = CIFAR100_URL
        self.data_md5 = CIFAR100_MD5
        self.flag = MODE_FLAG_MAP[self.mode + '100']