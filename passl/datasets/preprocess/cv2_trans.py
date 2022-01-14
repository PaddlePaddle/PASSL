# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
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


from __future__ import absolute_import
import math
import cv2
import random
import numpy as np
import types
from PIL import Image
import paddle
from paddle.vision.transforms import CenterCrop
from paddle.vision.transforms import crop, resize, hflip
from paddle.vision.transforms.transforms import _get_image_size

from . import cv2_func as F


_cv2_str_to_interpolation = {
    'nearest': cv2.INTER_NEAREST,
    'linear': cv2.INTER_LINEAR,
    'area': cv2.INTER_AREA,
    'cubic': cv2.INTER_CUBIC,
    'lanczos4': cv2.INTER_LANCZOS4,
}


class Compose(object):
    def __init__(self, trans):
        self.trans = trans

    def __call__(self, img):
        for t in self.trans:
            img = t(img)
        return img


class ByolNormalize(object):
    def __init__(self, mean=None, std=None):
        if mean is None or std is None:
            self.mean = None
            self.std = None
        else:
            # HWC
            self.mean = np.array(mean, dtype='float32').reshape([1, 1, 3])
            self.std = np.array(std, dtype='float32').reshape([1, 1, 3])

    def __call__(self, img):
        img = np.array(img)
        return F.normalize(img, self.mean, self.std)


class Resize(object):
    """ size, int or (h, w)"""
    def __init__(self, size, interpolation='linear'):
        assert isinstance(size, int) or len(size) == 2
        self.size = size
        self.interpolation = _cv2_str_to_interpolation[interpolation]

    def __call__(self, img):
        return F.resize(img, self.size, self.interpolation)

class ToCHW(object):
    def __call__(self, img):
        return F.to_chw(img)


class ByolToRGB(object):
    def __call__(self, img):
        return F.to_rgb_bgr(img)


class Lambda(object):
    def __init__(self, lambd):
        assert isinstance(lambd, types.LambdaType)
        self.lambd = lambd

    def __call__(self, img):
        return self.lambd(img)


class RandomTransforms(object):
    def __init__(self, trans):
        assert isinstance(trans, (list, tuple))
        self.trans = trans

    def __call__(self, *args, **kwargs):
        raise NotImplementedError()


class RandomApply(RandomTransforms):
    def __init__(self, trans, p=0.5):
        super(RandomApply, self).__init__(trans)
        self.p = p

    def __call__(self, img):
        if self.p < random.random():
            return img
        for t in self.trans:
            img = t(img)
        return img


class ByolRandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return F.hflip(img)
        return img


class ByolRandomVerticalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        if np.random.uniform() <= self.p:
            return F.vflip(img)
        return img


class ByolRandomResizedCrop(object):
    def __init__(self,
                 size,
                 scale=(0.08, 1.0),
                 ratio=(3./4., 4./3.),
                 interpolation='linear'):
        if type(size) is int:
            self.size = (size, size)
        else:
            self.size = size  # (h, w)
        self.scale = scale
        self.ratio = ratio
        self.interpolation = _cv2_str_to_interpolation[interpolation]

    def __call__(self, img):
        return F.random_crop_with_resize(
            img, self.size, self.scale, self.ratio, self.interpolation)


class ByolColorJitter(object):
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = [- brightness, brightness]
        self.contrast = [1 - contrast, 1 + contrast]
        self.saturation = [1 - saturation, 1 + saturation]
        self.hue = [-hue, hue]

    @staticmethod
    def get_params(brightness, contrast, saturation, hue):
        transforms = []

        brightness_factor = np.random.uniform(brightness[0], brightness[1])
        contrast_factor = np.random.uniform(contrast[0], contrast[1])
        saturation_factor = np.random.uniform(saturation[0], saturation[1])
        hue_factor = np.random.uniform(hue[0], hue[1])

        transforms.append(Lambda(lambda img: F.adjust_brightness(img, brightness_factor)))
        transforms.append(Lambda(lambda img: F.adjust_contrast(img, contrast_factor)))
        transforms.append(Lambda(lambda img: F.adjust_saturation(img, saturation_factor)))
        transforms.append(Lambda(lambda img: F.adjust_hue(img, hue_factor)))

        random.shuffle(transforms)
        transform = Compose(transforms)
        return transform

    def __call__(self, img):
        transform = self.get_params(self.brightness, self.contrast,
                                    self.saturation, self.hue)
        return transform(img)


class ByolRandomGrayscale(object):
    def __init__(self, p=0.1):
        self.p = p

    def __call__(self, img):
        if np.random.uniform() <= self.p:
            return F.to_grayscale(img)
        return img

class ByolCenterCrop(object):
    def __init__(self):
        pass
    
    def __call__(self,img):
        image_width, image_height = img.size
        padded_center_crop_size = int((224 / (224 + 32)) *np.minimum(image_height, image_width))
        return CenterCrop(size=padded_center_crop_size)(img)

class ByolRandomCrop(object):
    def __init__(self):
        pass

    def decode_and_random_crop(self,img):
        """Make a random crop of 224."""
        img_size = img.size
        area = img_size[1] * img_size[0]
        target_area = np.random.uniform( 0.08, 1.0) * area

        log_ratio = (math.log(3 / 4), math.log(4 / 3))
        aspect_ratio = math.exp(np.random.uniform(*log_ratio))

        w = int(np.round(np.sqrt(target_area * aspect_ratio)))
        h = int(np.round(np.sqrt(target_area / aspect_ratio)))

        w = np.minimum(w, img_size[0])
        h = np.minimum(h, img_size[1])

        offset_w = int(np.random.uniform(0,img_size[0] - w + 1))
        offset_h = int(np.random.uniform(0,img_size[1] - h + 1))
        return img.crop([offset_w,offset_h, w + offset_w, h + offset_h])

    def __call__(self,img):
        return self.decode_and_random_crop(img)


class RandomHorizontalFlipCoord(object):
    """Horizontally flip the given PIL Image randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, coord):
        """
        Args:
            img (PIL Image): Image to be flipped.

        Returns:
            PIL Image: Randomly flipped image.
        """
        if random.random() < self.p:
            coord_new = coord.clone()
            coord_new[0] = coord[2]
            coord_new[2] = coord[0]
            return hflip(img), coord_new
        return img, coord

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)

    

class RandomResizedCropCoord(object):
    """Crop the given PIL Image to random size and aspect ratio.

    A crop of random size (default: of 0.08 to 1.0) of the original size and a random
    aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio is made. This crop
    is finally resized to given size.
    This is popularly used to train the Inception networks.

    Args:
        size: expected output size of each edge
        scale: range of size of the origin size cropped
        ratio: range of aspect ratio of the origin aspect ratio cropped
        interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), interpolation='bilinear'):
        if isinstance(size, (tuple, list)):
            self.size = size
        else:
            self.size = (size, size)
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("range should be of kind (min, max)")

        self.interpolation = interpolation
        self.scale = scale
        self.ratio = ratio

    @staticmethod
    def get_params(img, scale, ratio):
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            img (PIL Image): Image to be cropped.
            scale (tuple): range of size of the origin size cropped
            ratio (tuple): range of aspect ratio of the origin aspect ratio cropped

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
                sized crop.
        """
        width, height = _get_image_size(img)
        area = height * width
        
        for attempt in range(0):
            target_area = random.uniform(*scale) * area
            log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                i = random.randint(0, height - h)
                j = random.randint(0, width - w)
                return i, j, h, w, height, width
        
        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if (in_ratio < min(ratio)):
            w = width
            h = int(round(w / min(ratio)))
        elif (in_ratio > max(ratio)):
            h = height
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w, height, width

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped and resized.

        Returns:
            PIL Image: Randomly cropped and resized image.
        """
        i, j, h, w, height, width = self.get_params(img, self.scale, self.ratio)
        coord = paddle.to_tensor([float(j) / (width - 1), float(i) / (height - 1),
                              float(j + w - 1) / (width - 1), float(i + h - 1) / (height - 1)])
        cropped_img = crop(img, i, j, h, w)
        return resize(cropped_img, self.size, self.interpolation), coord
        
