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


from paddle.vision.transforms import (
    Compose,
    Transpose,
    ColorJitter,
    RandomResizedCrop,
    RandomHorizontalFlip,
)
from passl.data.dataset.imagefolder_dataset import ImageFolder
from passl.data.preprocess import (
    RandomApply,
    GaussianBlur,
    NormalizeImage,
    RandomGrayscale,
)


class MultiCropDataset(ImageFolder):
    def __init__(self,
                 dataroot,
                 size_crops,
                 num_crops,
                 min_scale_crops,
                 max_scale_crops,
                 return_label=False):
        super(MultiCropDataset, self).__init__(dataroot)

        assert len(size_crops) == len(num_crops)
        assert len(min_scale_crops) == len(num_crops)
        assert len(max_scale_crops) == len(num_crops)
        self.return_label = return_label

        color_transform = [get_color_distortion(), get_pil_gaussian_blur()]
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        trans = []
        for i in range(len(size_crops)):
            randomresizedcrop = RandomResizedCrop(
                size_crops[i],
                scale=(min_scale_crops[i], max_scale_crops[i]),
            )
            trans.extend([Compose([
                randomresizedcrop,
                RandomHorizontalFlip(prob=0.5),
                Compose(color_transform),
                Transpose(),
                NormalizeImage(scale='1.0/255.0', mean=mean, std=std)])
            ] * num_crops[i])
        self.trans = trans

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        sample = list(map(lambda trans: trans(sample), self.trans))
        if self.return_label:
            return sample, target

        return sample


def get_pil_gaussian_blur(p=0.5):
    gaussian_blur = GaussianBlur(sigma=[.1, 2.], _PIL=True)
    rnd_gaussian_blur = RandomApply([gaussian_blur], p=p)
    return rnd_gaussian_blur


def get_color_distortion(s=1.0):
    # s is the strength of color distortion.
    color_jitter = ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
    rnd_color_jitter = RandomApply([color_jitter], p=0.8)
    rnd_gray = RandomGrayscale(p=0.2)
    color_distort = Compose([rnd_color_jitter, rnd_gray])
    return color_distort