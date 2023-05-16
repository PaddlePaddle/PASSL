# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

from passl.utils import logger
from passl.data.dataset import default_loader
from passl.data.dataset import ImageFolder

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif",
                  ".tiff", ".webp")


class FewShotDataset(ImageFolder):
    """
    This class inherits from :class:`~passl.data.datasets.ImageFolder`, so
    the dataset takes txt files containing image names to find the data
    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an numpy image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
        txt_file_name(string): The name of the txt file.
    """

    def __init__(self,
                 root,
                 transform=None,
                 target_transform=None,
                 loader=default_loader,
                 extensions=IMG_EXTENSIONS,
                 txt_file_name=None):
        super(FewShotDataset, self).__init__(root=root, transform=transform,
                                             target_transform=target_transform, loader=loader,
                                             extensions=extensions)

        assert txt_file_name is not None, "The txt_file_name should not be assigned."
        if os.path.isfile(txt_file_name):
            with open(txt_file_name, 'r') as f:
                list_imgs = [li.split('\n')[0] for li in f.readlines()]

            self.imgs = [(os.path.join(root, li.split('_')[0], li), self.class_to_idx[li.split('_')[0]]) for li in list_imgs]
        else:
            raise FileNotFoundError('{} is not existed'.format(txt_file_name))
        print('Previous information is not correct.')
        print(f'Actually, we have total {len(self.imgs)} images in semi-training setting.')
