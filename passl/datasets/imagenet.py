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
from .folder import DatasetFolder

from .preprocess import build_transforms
from .builder import DATASETS
from ..utils.misc import accuracy


@DATASETS.register()
class ImageNet(DatasetFolder):
    cls_filter = None

    def __init__(self,
                 dataroot,
                 return_label,
                 return_two_sample=False,
                 transforms=None,
                 view_trans1=None,
                 view_trans2=None):
        super(ImageNet, self).__init__(dataroot, cls_filter=self.cls_filter)

        self.return_label = return_label
        self.return_two_sample = return_two_sample
        self.transforms = transforms
        if transforms is not None:
            self.transform = build_transforms(transforms)
        if self.return_two_sample:
            self.view_transform1 = build_transforms(view_trans1)
            self.view_transform2 = build_transforms(view_trans2)


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
            sample1 = self.view_transform1(sample1)
            sample2 = self.view_transform2(sample2)
            if self.return_label:
                return sample1, sample2, target
            else:
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

@DATASETS.register()
class ImageNet100(ImageNet):
    cls_filter = lambda self, a: (a in [
            'n02869837', 'n01749939', 'n02488291', 'n02107142', 'n13037406', 'n02091831', 
            'n04517823', 'n04589890', 'n03062245', 'n01773797', 'n01735189', 'n07831146', 'n07753275', 
            'n03085013', 'n04485082', 'n02105505', 'n01983481', 'n02788148', 'n03530642', 'n04435653', 
            'n02086910', 'n02859443', 'n13040303', 'n03594734', 'n02085620', 'n02099849', 'n01558993', 
            'n04493381', 'n02109047', 'n04111531', 'n02877765', 'n04429376', 'n02009229', 'n01978455', 
            'n02106550', 'n01820546', 'n01692333', 'n07714571', 'n02974003', 'n02114855', 'n03785016', 
            'n03764736', 'n03775546', 'n02087046', 'n07836838', 'n04099969', 'n04592741', 'n03891251', 
            'n02701002', 'n03379051', 'n02259212', 'n07715103', 'n03947888', 'n04026417', 'n02326432', 
            'n03637318', 'n01980166', 'n02113799', 'n02086240', 'n03903868', 'n02483362', 'n04127249', 
            'n02089973', 'n03017168', 'n02093428', 'n02804414', 'n02396427', 'n04418357', 'n02172182', 
            'n01729322', 'n02113978', 'n03787032', 'n02089867', 'n02119022', 'n03777754', 'n04238763', 
            'n02231487', 'n03032252', 'n02138441', 'n02104029', 'n03837869', 'n03494278', 'n04136333', 
            'n03794056', 'n03492542', 'n02018207', 'n04067472', 'n03930630', 'n03584829', 'n02123045', 
            'n04229816', 'n02100583', 'n03642806', 'n04336792', 'n03259280', 'n02116738', 'n02108089', 
            'n03424325', 'n01855672', 'n02090622'
    ])

