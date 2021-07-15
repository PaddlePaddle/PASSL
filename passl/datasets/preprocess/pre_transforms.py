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

import random
from PIL import ImageFilter,Image
import numpy as np
import paddle
import cv2

import paddle.vision.transforms as PT
import paddle.vision.transforms.functional as F

from .builder import TRANSFORMS, build_transform

TRANSFORMS.register(PT.RandomResizedCrop)
TRANSFORMS.register(PT.ColorJitter)
TRANSFORMS.register(PT.Transpose)
TRANSFORMS.register(PT.Normalize)
TRANSFORMS.register(PT.RandomHorizontalFlip)
TRANSFORMS.register(PT.Resize)
TRANSFORMS.register(PT.CenterCrop)
TRANSFORMS.register(PT.ToTensor)

@TRANSFORMS.register()
class NormToOne():
    def __init__(self):
        pass

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be converted to grayscale.

        Returns:
            PIL Image: Randomly grayscaled image.
        """
        #norm_img = (img/255.).astype('float32')
        norm_img = Image.eval(img,lambda x:x/255.0)
        return norm_img

#add for BYOL
@TRANSFORMS.register()
class BrightnessTransform():
    def __init__(self,brightness):
        self.brightness = brightness

    def __call__(self,img):
        brightness_factor = np.random.uniform(-self.brightness,self.brightness)
        return F.adjust_brightness(img, brightness_factor)


@TRANSFORMS.register()
class ContrastTransform():
    def __init__(self,contrast):
        self.contrast = contrast

    def __call__(self,img):
        contrast_factor = np.random.uniform(1-self.contrast,1+self.contrast)
        return F.adjust_contrast(img, contrast_factor)

@TRANSFORMS.register()
class SaturationTransform():
    def __init__(self,saturation):
        self.saturation = saturation

    def __call__(self,img):
        saturation_factor = np.random.uniform(1-self.saturation,1+self.saturation)
        return F.adjust_saturation(img, saturation_factor)

@TRANSFORMS.register()
class HueTransform():
    def __init__(self,hue):
        self.hue = hue

    def __call__(self,img):
        hue_factor = np.random.uniform(-self.hue,self.hue)
        return F.adjust_hue(img, hue_factor)


@TRANSFORMS.register()
class Clip():
    def __init__(self,min_val=0.0,max_val=1.0):
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be converted to grayscale.

        Returns:
            PIL Image: Cliped image.
        """
        clip_img = Image.eval(img,lambda x:max(0,min(x,1)))
        return clip_img

@TRANSFORMS.register()
class RandomApply():
    """Apply randomly a list of transformations with a given probability

    Args:
        transforms (list or tuple): list of transformations
        p (float): probability
    """

    def __init__(self, transforms, p=0.5):
        _transforms = []
        if isinstance(transforms, (list, tuple)):
            for transform in transforms:
                if isinstance(transform, dict):
                    _transforms.append(build_transform(transform))
                else:
                    _transforms.append(transform)

        self.transforms = _transforms
        self.p = p

    def __call__(self, img):
        if self.p < random.random():
            return img
        for t in self.transforms:
            img = t(img)
        return img


@TRANSFORMS.register()
class RandomGrayscale(object):
    """Randomly convert image to grayscale with a probability of p (default 0.1).

    Args:
        p (float): probability that image should be converted to grayscale.

    Returns:
        PIL Image: Grayscale version of the input image with probability p and unchanged
        with probability (1-p).
        - If input image is 1 channel: grayscale version is 1 channel
        - If input image is 3 channel: grayscale version is 3 channel with r == g == b

    """

    def __init__(self, p=0.1):
        self.p = p

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be converted to grayscale.

        Returns:
            PIL Image: Randomly grayscaled image.
        """
        num_output_channels = 1 if img.mode == 'L' else 3
        if random.random() < self.p:
            return F.to_grayscale(img, num_output_channels=num_output_channels)
        return img


@TRANSFORMS.register()
class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self,kernel_size, sigma=[.1, 2.],use_cv=True):
        self.sigma = sigma
        self.kernel_size = kernel_size
        self.use_cv = use_cv

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        if self.use_cv:
            x = cv2.GaussianBlur(np.array(x), (self.kernel_size, self.kernel_size), sigma)
            return Image.fromarray(x.astype(np.uint8))
        else:
            x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
            return x


@TRANSFORMS.register()
class Solarization(object):
    """Solarization augmentation in BYOL https://arxiv.org/abs/2006.07733."""

    def __init__(self, threshold=128,norm=False):
        self.threshold = threshold
        self.norm = norm

    def __call__(self, img):
        img = np.array(img)
        if not self.norm:
            img = np.where(img < self.threshold, img, 255 -img)
        else:
            img = np.where(img < self.threshold, img, 1 -img)
        return Image.fromarray(img.astype(np.uint8))

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str
