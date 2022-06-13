""" Transforms Factory
Factory methods for building image transforms for use with TIMM (PyTorch Image Models)

Hacked together by / Copyright 2020 Ross Wightman
"""
import random
import numpy as np
from PIL import Image

import paddle.vision.transforms.functional as F
from paddle.vision import transforms

from .auto_augment import rand_augment_transform
from ..constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from ..random_erasing import RandomErasing

_RANDOM_INTERPOLATION = ('bilinear', 'bicubic')


class WithLabel(object):
    def _apply_label(self, label):
        return label


class LVViTRandomHorizontalFlip(WithLabel, transforms.RandomHorizontalFlip):
    def __init__(self, *arg, **kwargs):
        kwargs['keys'] = ("image", "label")
        super().__init__(*arg, **kwargs)

    def _apply_label(self, label):
        return label[:, :, :, ::-1]


class LVViTRandomVerticalFlip(WithLabel, transforms.RandomVerticalFlip):
    def __init__(self, *arg, **kwargs):
        kwargs['keys'] = ("image", "label")
        super().__init__(*arg, **kwargs)

    def _apply_label(self, label):
        return label[:, :, ::-1]


class LVViTRandomResizedCropAndInterpolation(WithLabel,
                                             transforms.RandomResizedCrop):
    def __init__(self, *arg, **kwargs):
        kwargs['keys'] = ("image", "label")
        super().__init__(*arg, **kwargs)

    def _get_params(self, inputs):
        image, label = inputs
        return self._get_param(image), transforms.transforms._get_image_size(
            image)

    def _apply_image(self, img):
        interpolation = self.interpolation
        if self.interpolation == 'random':
            interpolation = random.choice(_RANDOM_INTERPOLATION)

        (i, j, h, w), (width, height) = self.params

        cropped_img = F.crop(img, i, j, h, w)
        return F.resize(cropped_img, self.size, interpolation)

    def _apply_label(self, label):
        (i, j, h, w), (width, height) = self.params

        coords = (i / height, j / width, h / height, w / width)
        coords_map = np.zeros_like(label[0:1])
        # trick to store coords_map is label
        coords_map[0, 0, 0, 0], coords_map[0, 0, 0, 1], \
            coords_map[0, 0, 0, 2], coords_map[0, 0, 0, 3] = coords
        return np.concatenate([label, coords_map])


class LVViTToTensor(WithLabel, transforms.ToTensor):
    def __init__(self, *arg, **kwargs):
        kwargs['keys'] = ("image", "label")
        super().__init__(*arg, **kwargs)


class LVViTNormalize(WithLabel, transforms.Normalize):
    def __init__(self, *arg, **kwargs):
        kwargs['keys'] = ("image", "label")
        super().__init__(*arg, **kwargs)


class LVViTRandomErasing(RandomErasing):
    def __call__(self, input):
        return super().__call__(input[0]), input[1]


def _pil_interp(method):
    if method == 'bicubic':
        return Image.BICUBIC
    elif method == 'lanczos':
        return Image.LANCZOS
    elif method == 'hamming':
        return Image.HAMMING
    else:
        # default bilinear, do we want to allow nearest?
        return Image.BILINEAR


class LVViTAutoAugment(WithLabel):
    def __init__(self,
                 config_str,
                 img_size,
                 interpolation,
                 mean=IMAGENET_DEFAULT_MEAN,
                 std=IMAGENET_DEFAULT_STD):
        super().__init__()
        assert isinstance(config_str, str)
        if isinstance(img_size, tuple):
            img_size_min = min(img_size)
        else:
            img_size_min = img_size
        aa_params = dict(
            translate_const=int(img_size_min * 0.45),
            img_mean=tuple([min(255, round(255 * x)) for x in mean]), )
        if interpolation and interpolation != 'random':
            aa_params['interpolation'] = _pil_interp(interpolation)
        if config_str.startswith('rand'):
            self.transform = rand_augment_transform(config_str, aa_params)
        else:
            raise NotImplementedError

    def __call__(self, data):
        return self.transform(data)
