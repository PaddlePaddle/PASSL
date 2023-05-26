# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import warnings
import numbers
from collections.abc import Sequence
from typing import Any, List, Optional, Tuple, Union

from functools import partial
import six
import math
import random
import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageOps
from paddle.vision.transforms import ColorJitter as PPColorJitter
from paddle.vision.transforms import ToTensor, Normalize

from passl.utils import logger

__all__ = [
    "Compose",
    "TwoViewsTransform",
    "ToTensor",
    "DecodeImage",
    "RandomApply",
    "ResizeImage",
    "Resize",
    "CenterCropImage",
    "CenterCrop",
    "RandCropImage",
    "RandomResizedCrop",
    "RandomResizedCropAndInterpolation",
    "RandomResizedCropWithTwoImages",
    "RandFlipImage",
    "RandomHorizontalFlip",
    "NormalizeImage",
    "Normalize",
    "ToCHWImage",
    "ColorJitter",
    "RandomErasing",
    "RandomGrayscale",
    "SimCLRGaussianBlur",
    "BYOLSolarize",
    "MAERandCropImage",
]


class OperatorParamError(ValueError):
    """ OperatorParamError
    """
    pass


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += f"    {t}"
        format_string += "\n)"
        return format_string


class TwoViewsTransform(object):
    """Take two random crops of one image"""

    def __init__(self, base_transform1, base_transform2):
        self.base_transform1 = base_transform1
        self.base_transform2 = base_transform2

    def __call__(self, x):
        im1 = self.base_transform1(x)
        im2 = self.base_transform2(x)
        return im1, im2


class DecodeImage(object):
    """ decode image """

    def __init__(self, to_rgb=True, channel_first=False):
        self.to_rgb = to_rgb
        self.channel_first = channel_first

    def __call__(self, img):
        if six.PY2:
            assert type(img) is str and len(
                img) > 0, "invalid input 'img' in DecodeImage"
        else:
            assert type(img) is bytes and len(
                img) > 0, "invalid input 'img' in DecodeImage"
        data = np.frombuffer(img, dtype='uint8')
        img = cv2.imdecode(data, 1)
        if self.to_rgb:
            assert img.shape[2] == 3, 'invalid shape of image[%s]' % (
                img.shape)
            img = img[:, :, ::-1]

        if self.channel_first:
            img = img.transpose((2, 0, 1))

        return img


def _is_pil_image(img):
    return isinstance(img, Image.Image)


_cv2_interp_from_str = {
    'nearest': cv2.INTER_NEAREST,
    'bilinear': cv2.INTER_LINEAR,
    'area': cv2.INTER_AREA,
    'bicubic': cv2.INTER_CUBIC,
    'lanczos': cv2.INTER_LANCZOS4
}

if hasattr(Image, "Resampling"):
    _pil_interp_from_str = {
        'nearest': Image.Resampling.NEAREST,
        'bilinear': Image.Resampling.BILINEAR,
        'bicubic': Image.Resampling.BICUBIC,
        'box': Image.Resampling.BOX,
        'lanczos': Image.Resampling.LANCZOS,
        'hamming': Image.Resampling.HAMMING
    }
else:
    _pil_interp_from_str = {
        'nearest': Image.NEAREST,
        'bilinear': Image.BILINEAR,
        'bicubic': Image.BICUBIC,
        'box': Image.BOX,
        'lanczos': Image.LANCZOS,
        'hamming': Image.HAMMING
    }


def _pil_resize(src, size, resample):
    pil_img = src
    if not _is_pil_image(pil_img):
        pil_img = Image.fromarray(pil_img)
    pil_img = pil_img.resize(size, resample)
    return pil_img


def resize(img, size, interpolation=None, backend="cv2"):
    assert backend.lower() in ['cv2', 'pil']
    if backend.lower() == "cv2":
        if isinstance(interpolation, str):
            interpolation = _cv2_interp_from_str[interpolation.lower()]
        # compatible with opencv < version 4.4.0
        elif interpolation is None:
            interpolation = cv2.INTER_LINEAR
        return cv2.resize(img, size, interpolation=interpolation)
    elif backend.lower() == "pil":
        if isinstance(interpolation, str):
            interpolation = _pil_interp_from_str[interpolation.lower()]
        return _pil_resize(img, size, resample=interpolation)


_RANDOM_INTERPOLATION = ('bilinear', 'bicubic')


class UnifiedResize(object):
    def __init__(self, interpolation=None, backend="cv2"):
        if interpolation == 'random':
            self.interpolation = _RANDOM_INTERPOLATION
        self.interpolation = interpolation
        self.backend = backend

    def __call__(self, img, size):
        interpolation = self.interpolation
        if isinstance(self.interpolation, (tuple, list)):
            interpolation = random.choice(self.interpolation)
        return resize(img, size, interpolation, self.backend)


class ResizeImage(object):
    """ resize image """

    def __init__(self,
                 size=None,
                 resize_short=None,
                 interpolation=None,
                 backend="cv2"):
        if resize_short is not None and resize_short > 0:
            self.resize_short = resize_short
            self.w = None
            self.h = None
        elif size is not None:
            self.resize_short = None
            self.w = size if type(size) is int else size[0]
            self.h = size if type(size) is int else size[1]
        else:
            raise OperatorParamError("invalid params for ReisizeImage for '\
                'both 'size' and 'resize_short' are None")

        self._resize_func = UnifiedResize(
            interpolation=interpolation, backend=backend)

    def __call__(self, img):
        _, img_h, img_w = get_dimensions(img)
        if self.resize_short is not None:
            percent = float(self.resize_short) / min(img_w, img_h)
            w = int(round(img_w * percent))
            h = int(round(img_h * percent))
        else:
            w = self.w
            h = self.h
        return self._resize_func(img, (w, h))


class Resize(object):
    def __init__(self,
                 size,
                 interpolation='bilinear',
                 max_size=None,
                 antialias=None,
                 backend="pil"):
        if not isinstance(size, (int, Sequence)):
            raise TypeError(
                f"Size should be int or sequence. Got {type(size)}")
        if isinstance(size, Sequence) and len(size) not in (1, 2):
            raise ValueError(
                "If size is a sequence, it should have 1 or 2 values")

        if isinstance(size, (list, tuple)):
            if len(size) not in [1, 2]:
                raise ValueError(
                    f"Size must be an int or a 1 or 2 element tuple/list, not a {len(size)} element tuple/list"
                )
            if max_size is not None and len(size) != 1:
                raise ValueError(
                    "max_size should only be passed if size specifies the length of the smaller edge, "
                    "i.e. size should be an int or a sequence of length 1 in torchscript mode."
                )

        if isinstance(size, int):
            size = [size]

        self.size = size
        self.max_size = max_size

        self.interpolation = interpolation
        self.antialias = antialias

        self._resize_func = UnifiedResize(
            interpolation=interpolation, backend=backend)

    def _compute_resized_output_size(
            self,
            image_size: Tuple[int, int],
            size: List[int],
            max_size: Optional[int]=None) -> List[int]:
        if len(size) == 1:  # specified size only for the smallest edge
            h, w = image_size
            short, long = (w, h) if w <= h else (h, w)
            requested_new_short = size if isinstance(size, int) else size[0]

            new_short, new_long = requested_new_short, int(
                requested_new_short * long / short)

            if max_size is not None:
                if max_size <= requested_new_short:
                    raise ValueError(
                        f"max_size = {max_size} must be strictly greater than the requested "
                        f"size for the smaller edge size = {size}")
                if new_long > max_size:
                    new_short, new_long = int(max_size * new_short /
                                              new_long), max_size

            new_w, new_h = (new_short, new_long) if w <= h else (new_long,
                                                                 new_short)
        else:  # specified both h and w
            new_w, new_h = size[1], size[0]
        return [new_h, new_w]

    def __call__(self, img):
        _, img_h, img_w = get_dimensions(img)

        h, w = self._compute_resized_output_size((img_h, img_w), self.size,
                                                 self.max_size)
        return self._resize_func(img, (w, h))


class CenterCropImage(object):
    """ crop image """

    def __init__(self, size):
        if type(size) is int:
            self.size = (size, size)
        else:
            self.size = size  # (h, w)

    def __call__(self, img):
        w, h = self.size
        _, img_h, img_w = get_dimensions(img)
        w_start = (img_w - w) // 2
        h_start = (img_h - h) // 2

        return crop(img, h_start, w_start, h, w)


class CenterCrop(object):
    """ center crop image, align torchvision """

    def __init__(self, size):
        self.size = self._setup_size(
            size,
            error_msg="Please provide only two dimensions (h, w) for size.")

    def _setup_size(self, size, error_msg):

        if isinstance(size, numbers.Number):
            return int(size), int(size)

        if isinstance(size, Sequence) and len(size) == 1:
            return size[0], size[0]

        if len(size) != 2:
            raise ValueError(error_msg)

        return size

    def __call__(self, img):
        _, image_height, image_width = get_dimensions(img)
        crop_height, crop_width = self.size

        if crop_width > image_width or crop_height > image_height:
            padding_ltrb = [
                (crop_width - image_width) // 2
                if crop_width > image_width else 0,
                (crop_height - image_height) // 2
                if crop_height > image_height else 0,
                (crop_width - image_width + 1) // 2
                if crop_width > image_width else 0,
                (crop_height - image_height + 1) // 2
                if crop_height > image_height else 0,
            ]
            # TODO(GuoxiaWang): implement pad function
            img = pad(img, padding_ltrb, fill=0)  # PIL uses fill value 0
            _, image_height, image_width = get_dimensions(img)
            if crop_width == image_width and crop_height == image_height:
                return img

        crop_top = int(round((image_height - crop_height) / 2.0))
        crop_left = int(round((image_width - crop_width) / 2.0))
        return crop(img, crop_top, crop_left, crop_height, crop_width)


class RandCropImage(object):
    """ random crop image """

    def __init__(self,
                 size,
                 scale=None,
                 ratio=None,
                 interpolation=None,
                 backend="cv2"):
        if type(size) is int:
            self.size = (size, size)  # (h, w)
        else:
            self.size = size

        self.scale = [0.08, 1.0] if scale is None else scale
        self.ratio = [3. / 4., 4. / 3.] if ratio is None else ratio

        self._resize_func = UnifiedResize(
            interpolation=interpolation, backend=backend)

    def __call__(self, img):
        size = self.size
        scale = self.scale
        ratio = self.ratio

        aspect_ratio = math.sqrt(random.uniform(*ratio))
        w = 1. * aspect_ratio
        h = 1. / aspect_ratio

        _, img_h, img_w = get_dimensions(img)

        bound = min((float(img_w) / img_h) / (w**2),
                    (float(img_h) / img_w) / (h**2))
        scale_max = min(scale[1], bound)
        scale_min = min(scale[0], bound)

        target_area = img_w * img_h * random.uniform(scale_min, scale_max)
        target_size = math.sqrt(target_area)
        w = int(target_size * w)
        h = int(target_size * h)

        i = random.randint(0, img_w - w)
        j = random.randint(0, img_h - h)

        img = img[j:j + h, i:i + w, :]

        return self._resize_func(img, size)


def _setup_size(size, error_msg):
    if isinstance(size, numbers.Number):
        return int(size), int(size)

    if isinstance(size, Sequence) and len(size) == 1:
        return size[0], size[0]

    if len(size) != 2:
        raise ValueError(error_msg)

    return size


def get_dimensions(img: Any) -> List[int]:
    if _is_pil_image(img):
        if hasattr(img, "getbands"):
            channels = len(img.getbands())
        else:
            channels = img.channels
        width, height = img.size
        return [channels, height, width]
    else:
        height, width = img.shape[:2]
        if len(img.shape) == 2:
            channels = 1
        else:
            channels = img.shape[-1]
        return [channels, height, width]


def crop(img, top: int, left: int, height: int, width: int):
    if not _is_pil_image(img):
        return img[top:top + height, left:left + width, :]

    return img.crop((left, top, left + width, top + height))


def resized_crop(
        img,
        top: int,
        left: int,
        height: int,
        width: int,
        size: List[int],
        interpolation: str="bilinear",
        antialias: Optional[bool]=None, ):
    img = crop(img, top, left, height, width)
    img = resize(img, size, interpolation=interpolation, backend="pil")
    return img


class RandomResizedCrop(object):
    def __init__(
            self,
            size,
            scale=(0.08, 1.0),
            ratio=(3.0 / 4.0, 4.0 / 3.0),
            interpolation="bilinear",
            antialias: Optional[bool]=None, ):
        super().__init__()
        self.size = _setup_size(
            size,
            error_msg="Please provide only two dimensions (h, w) for size.")

        if not isinstance(scale, Sequence):
            raise TypeError("Scale should be a sequence")
        if not isinstance(ratio, Sequence):
            raise TypeError("Ratio should be a sequence")
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("Scale and ratio should be of kind (min, max)")

        self.interpolation = interpolation
        self.antialias = antialias
        self.scale = scale
        self.ratio = ratio

    @staticmethod
    def get_params(img, scale: List[float],
                   ratio: List[float]) -> Tuple[int, int, int, int]:
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            img (PIL Image or Tensor): Input image.
            scale (list): range of scale of the origin size cropped
            ratio (list): range of aspect ratio of the origin aspect ratio cropped

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
            sized crop.
        """
        _, height, width = get_dimensions(img)
        area = height * width

        log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
        for _ in range(10):
            target_area = area * random.uniform(scale[0], scale[1])
            aspect_ratio = math.exp(random.uniform(log_ratio[0], log_ratio[1]))

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                i = random.randint(0, height - h + 1)
                j = random.randint(0, width - w + 1)
                return i, j, h, w

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if in_ratio < min(ratio):
            w = width
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = height
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w

    def __call__(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped and resized.

        Returns:
            PIL Image or Tensor: Randomly cropped and resized image.
        """
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        return resized_crop(
            img,
            i,
            j,
            h,
            w,
            self.size,
            self.interpolation,
            antialias=self.antialias)


class RandomResizedCropWithTwoImages(RandomResizedCrop):
    def __init__(
            self,
            size,
            second_size=None,
            scale=(0.08, 1.0),
            ratio=(3. / 4., 4. / 3.),
            interpolation='bilinear',
            second_interpolation='lanczos',
            antialias: Optional[bool]=None, ):
        super().__init__(
            size,
            scale=scale,
            ratio=ratio,
            interpolation=interpolation,
            antialias=antialias)

        self.second_size = None
        if second_size is not None:
            self.second_size = _setup_size(
                second_size,
                error_msg="Please provide only two dimensions (h, w) for size.")

        self.second_interpolation = second_interpolation

    def __call__(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped and resized.

        Returns:
            PIL Image or Tensor: Randomly cropped and resized image.
        """
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        if self.second_size is not None:
            img1 = resized_crop(
                img,
                i,
                j,
                h,
                w,
                self.size,
                self.interpolation,
                antialias=self.antialias)
            img2 = resized_crop(
                img,
                i,
                j,
                h,
                w,
                self.second_size,
                self.second_interpolation,
                antialias=self.antialias)
            return img1, img2

        else:
            return resized_crop(
                img,
                i,
                j,
                h,
                w,
                self.size,
                self.interpolation,
                antialias=self.antialias)


class RandomResizedCropAndInterpolation(RandCropImage):
    """ only rename """
    pass


class MAERandCropImage(RandCropImage):
    """
    RandomResizedCrop for matching TF/TPU implementation: no for-loop is used.
    This may lead to results different with torchvision's version.
    Following BYOL's TF code:
    https://github.com/deepmind/deepmind-research/blob/master/byol/utils/dataset.py#L206
    """

    def __call__(self, img):
        size = self.size

        _, img_h, img_w = get_dimensions(img)

        target_area = img_w * img_h * np.random.uniform(*self.scale)
        log_ratio = tuple(math.log(x) for x in self.ratio)
        aspect_ratio = math.exp(np.random.uniform(*log_ratio))

        w = int(round(math.sqrt(target_area * aspect_ratio)))
        h = int(round(math.sqrt(target_area / aspect_ratio)))

        w = min(w, img_w)
        h = min(h, img_h)

        i = random.randint(0, img_w - w)
        j = random.randint(0, img_h - h)

        img = crop(img, j, i, h, w)
        return self._resize_func(img, size)


class RandFlipImage(object):
    """ random flip image
        flip_code:
            1: Flipped Horizontally
            0: Flipped Vertically
            -1: Flipped Horizontally & Vertically
    """

    def __init__(self, flip_code=1):
        assert flip_code in [-1, 0, 1
                             ], "flip_code should be a value in [-1, 0, 1]"
        self.flip_code = flip_code

    def __call__(self, img):
        if random.randint(0, 1) == 1:
            # backward compatibility
            if _is_pil_image(img):
                if self.flip_code == 1:
                    return img.transpose(0)
                elif self.flip_code == 0:
                    return img.transpose(1)
                else:
                    raise ValueError(
                        "PIL.Image does not support Flipped Horizontally & Vertically"
                    )
            else:
                return cv2.flip(img, self.flip_code)
        else:
            return img


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        if np.random.rand() < self.p:
            return img.transpose(0)  # FLIP_LEFT_RIGHT = 0, FLIP_TOP_BOTTOM = 1
        else:
            return img


class NormalizeImage(object):
    """ normalize image such as substract mean, divide std
    """

    def __init__(self,
                 scale=None,
                 mean=None,
                 std=None,
                 order='chw',
                 output_fp16=False,
                 channel_num=3):
        if isinstance(scale, str):
            scale = eval(scale)
        assert channel_num in [
            3, 4
        ], "channel number of input image should be set to 3 or 4."
        self.channel_num = channel_num
        self.output_dtype = 'float16' if output_fp16 else 'float32'
        self.scale = np.float32(scale if scale is not None else 1.0 / 255.0)
        self.order = order
        mean = mean if mean is not None else [0.485, 0.456, 0.406]
        std = std if std is not None else [0.229, 0.224, 0.225]

        shape = (3, 1, 1) if self.order == 'chw' else (1, 1, 3)
        self.mean = np.array(mean).reshape(shape).astype('float32')
        self.std = np.array(std).reshape(shape).astype('float32')

    def __call__(self, img):
        if _is_pil_image(img):
            img = np.array(img)

        assert isinstance(img,
                          np.ndarray), "invalid input 'img' in NormalizeImage"

        img = (img.astype('float32') * self.scale - self.mean) / self.std

        if self.channel_num == 4:
            img_h = img.shape[1] if self.order == 'chw' else img.shape[0]
            img_w = img.shape[2] if self.order == 'chw' else img.shape[1]
            pad_zeros = np.zeros(
                (1, img_h, img_w)) if self.order == 'chw' else np.zeros(
                    (img_h, img_w, 1))
            img = (np.concatenate(
                (img, pad_zeros), axis=0)
                   if self.order == 'chw' else np.concatenate(
                       (img, pad_zeros), axis=2))
        return img.astype(self.output_dtype)


class ToCHWImage(object):
    """ convert hwc image to chw image
    """

    def __init__(self):
        pass

    def __call__(self, img):
        if _is_pil_image(img):
            img = np.array(img)

        return img.transpose((2, 0, 1))


class ColorJitter(PPColorJitter):
    """ColorJitter.
    """

    def __init__(self, p=1.0, *args, **kwargs):
        self.p = p
        super().__init__(*args, **kwargs)

    def __call__(self, img):
        if random.random() < self.p:
            if not _is_pil_image(img):
                img = np.ascontiguousarray(img)
                img = Image.fromarray(img)
                img = super()._apply_image(img)
                img = np.asarray(img)
            else:
                img = super()._apply_image(img)
        return img


class Pixels(object):
    def __init__(self, mode="const", mean=[0., 0., 0.]):
        self._mode = mode
        self._mean = mean

    def __call__(self, h=224, w=224, c=3):
        if self._mode == "rand":
            return np.random.normal(size=(1, 1, 3))
        elif self._mode == "pixel":
            return np.random.normal(size=(h, w, c))
        elif self._mode == "const":
            return self._mean
        else:
            raise Exception(
                "Invalid mode in RandomErasing, only support \"const\", \"rand\", \"pixel\""
            )


class RandomErasing(object):
    """RandomErasing.
    This code is adapted from https://github.com/zhunzhong07/Random-Erasing, and refer to Timm.
    """

    def __init__(self,
                 EPSILON=0.5,
                 sl=0.02,
                 sh=0.4,
                 r1=0.3,
                 mean=[0., 0., 0.],
                 attempt=100,
                 use_log_aspect=False,
                 mode='const'):
        self.EPSILON = eval(EPSILON) if isinstance(EPSILON, str) else EPSILON
        self.sl = eval(sl) if isinstance(sl, str) else sl
        self.sh = eval(sh) if isinstance(sh, str) else sh
        r1 = eval(r1) if isinstance(r1, str) else r1
        self.r1 = (math.log(r1), math.log(1 / r1)) if use_log_aspect else (
            r1, 1 / r1)
        self.use_log_aspect = use_log_aspect
        self.attempt = attempt
        self.get_pixels = Pixels(mode, mean)

    def __call__(self, img):
        if random.random() > self.EPSILON:
            return img

        for _ in range(self.attempt):
            area = img.shape[0] * img.shape[1]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(*self.r1)
            if self.use_log_aspect:
                aspect_ratio = math.exp(aspect_ratio)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.shape[1] and h < img.shape[0]:
                pixels = self.get_pixels(h, w, img.shape[2])
                x1 = random.randint(0, img.shape[0] - h)
                y1 = random.randint(0, img.shape[1] - w)
                if img.shape[2] == 3:
                    img[x1:x1 + h, y1:y1 + w, :] = pixels
                else:
                    img[x1:x1 + h, y1:y1 + w, 0] = pixels[0]
                return img
        return img


class RandomApply(object):
    def __init__(self, transforms, p=0.5):
        self.transforms = transforms
        self.p = p

    def __call__(self, img):
        if self.p < np.random.rand():
            return img
        for t in self.transforms:
            img = t(img)
        return img


class RandomGrayscale(object):
    def __init__(self, p=0.1):
        self.p = p

    def __call__(self, img):
        num_output_channels, _, _ = get_dimensions(img)

        if np.random.rand() < self.p:

            if not _is_pil_image(img):
                img = np.ascontiguousarray(img)
                img = Image.fromarray(img)

                if num_output_channels == 1:
                    img = img.convert("L")
                    img = np.array(img, dtype=np.uint8)
                elif num_output_channels == 3:
                    img = img.convert("L")
                    img = np.array(img, dtype=np.uint8)
                    img = np.dstack([img, img, img])
                else:
                    raise ValueError(
                        "num_output_channels should be either 1 or 3")
            else:
                if num_output_channels == 1:
                    img = img.convert("L")
                elif num_output_channels == 3:
                    img = img.convert("L")
                    np_img = np.array(img, dtype=np.uint8)
                    np_img = np.dstack([np_img, np_img, np_img])
                    img = Image.fromarray(np_img, "RGB")
                else:
                    raise ValueError(
                        "num_output_channels should be either 1 or 3")
        return img


class SimCLRGaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.], p=1.0):
        self.p = p
        self.sigma = sigma

    def __call__(self, img):
        if random.random() < self.p:
            sigma = random.uniform(self.sigma[0], self.sigma[1])
            if not _is_pil_image(img):
                img = np.ascontiguousarray(img)
                img = Image.fromarray(img)
                img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
                img = np.asarray(img)
            else:
                img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
        return img


class BYOLSolarize(object):
    """Solarize augmentation from BYOL: https://arxiv.org/abs/2006.07733"""

    def __init__(self, p=1.0):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            if not _is_pil_image(img):
                img = np.ascontiguousarray(img)
                img = Image.fromarray(img)
                img = ImageOps.solarize(img)
                img = np.asarray(img)
            else:
                img = ImageOps.solarize(img)
        return img
