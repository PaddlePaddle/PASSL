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

import math
import random
from PIL import ImageFilter, Image, ImageOps
import cv2
import numpy as np
from functools import partial

import paddle
import paddle.vision.transforms as PT
import paddle.vision.transforms.functional as F

from .mixup import Mixup
from .builder import TRANSFORMS, build_transform
from .random_erasing import RandomErasing
from .constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, DEFAULT_CROP_PCT
from .auto_augment import rand_augment_transform, augment_and_mix_transform, auto_augment_transform
from .cv2_trans import ByolRandomHorizontalFlip, ByolColorJitter, ByolRandomGrayscale, ByolNormalize, \
         ToCHW, ByolToRGB, ByolCenterCrop, ByolRandomCrop
from .img_pil_pixpro_transforms import RandomResizedCropCoord, RandomHorizontalFlipCoord

TRANSFORMS.register(PT.RandomResizedCrop)
TRANSFORMS.register(PT.ColorJitter)
TRANSFORMS.register(PT.Transpose)
TRANSFORMS.register(PT.Normalize)
TRANSFORMS.register(PT.RandomHorizontalFlip)
TRANSFORMS.register(PT.Resize)
TRANSFORMS.register(PT.CenterCrop)
TRANSFORMS.register(PT.ToTensor)

TRANSFORMS.register(ByolRandomHorizontalFlip)
TRANSFORMS.register(ByolColorJitter)
TRANSFORMS.register(ByolRandomGrayscale)
TRANSFORMS.register(ByolNormalize)
TRANSFORMS.register(ToCHW)
TRANSFORMS.register(ByolToRGB)
TRANSFORMS.register(ByolRandomCrop)
TRANSFORMS.register(ByolCenterCrop)

TRANSFORMS.register(RandomErasing)
TRANSFORMS.register(Mixup)

TRANSFORMS.register(RandomResizedCropCoord)
TRANSFORMS.register(RandomHorizontalFlipCoord)

@TRANSFORMS.register()
class Clip():
    def __init__(self, min_val=0.0, max_val=1.0):
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be converted to grayscale.

        Returns:
            PIL Image: Cliped image.
        """
        clip_img = img.clip(self.min_val, self.max_val)
        return clip_img


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
        norm_img = (img / 255.).astype('float32')
        return norm_img


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
    def __init__(self, sigma=[.1, 2.], _PIL=False):
        self.sigma = sigma
        self.kernel_size = 23
        self._PIL = _PIL

    def __call__(self, x):
        sigma = np.random.uniform(self.sigma[0], self.sigma[1])
        if self._PIL:
            x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
            return x
        else:
            import cv2
            x = cv2.GaussianBlur(np.array(x),
                                 (self.kernel_size, self.kernel_size), sigma)
            return Image.fromarray(x.astype(np.uint8))


@TRANSFORMS.register()
class Solarization(object):
    def __init__(self, threshold=128):
        self.threshold = threshold

    def __call__(self, sample):
        return ImageOps.solarize(sample, self.threshold)


@TRANSFORMS.register()
class ToRGB(object):
    def __init__(self, mode='RGB', return_type='pil'):
        self.mode = mode
        self.return_type = return_type
        assert return_type in ['pil', 'numpy']

    def __call__(self, sample):
        if isinstance(sample, Image.Image) and sample.mode != self.mode:
            sample = sample.convert(self.mode)
        if isinstance(sample, np.ndarray):
            sample = sample[..., ::-1]
        if self.return_type == 'numpy' and not isinstance(sample, np.ndarray):
            sample = np.asarray(sample)
        if self.return_type == 'pil' and not isinstance(sample, Image.Image):
            sample = Image.fromarray(sample)
        return sample


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


@TRANSFORMS.register()
class AutoAugment(PT.BaseTransform):
    def __init__(self,
                 config_str,
                 img_size,
                 interpolation,
                 mean=IMAGENET_DEFAULT_MEAN,
                 std=IMAGENET_DEFAULT_STD,
                 keys=None):
        super(AutoAugment, self).__init__(keys)
        assert isinstance(config_str, str)
        if isinstance(img_size, tuple):
            img_size_min = min(img_size)
        else:
            img_size_min = img_size
        aa_params = dict(
            translate_const=int(img_size_min * 0.45),
            img_mean=tuple([min(255, round(255 * x)) for x in mean]),
        )
        if interpolation and interpolation != 'random':
            aa_params['interpolation'] = _pil_interp(interpolation)
        if config_str.startswith('rand'):
            self.transform = rand_augment_transform(config_str, aa_params)
        elif config_str.startswith('augmix'):
            aa_params['translate_pct'] = 0.3
            self.transform = augment_and_mix_transform(config_str, aa_params)
        elif config_str == '':
            self.transform = None
        else:
            self.transform = auto_augment_transform(config_str, aa_params)

    def _apply_image(self, img):
        if self.transform != None:
            is_pil = isinstance(img, Image.Image)
            if not is_pil:
                img = np.ascontiguousarray(img)
                img = Image.fromarray(img)
            img = self.transform(img)
            if not is_pil:
                img = np.asarray(img)
        return img
    

class UnifiedResize(object):
    """
        https://github.com/PaddlePaddle/PaddleClas/blob/release/2.3/ppcls/data/preprocess/ops/operators.py
    """
    def __init__(self, interpolation=None, backend="cv2"):
        _cv2_interp_from_str = {
            'nearest': cv2.INTER_NEAREST,
            'bilinear': cv2.INTER_LINEAR,
            'area': cv2.INTER_AREA,
            'bicubic': cv2.INTER_CUBIC,
            'lanczos': cv2.INTER_LANCZOS4
        }
        _pil_interp_from_str = {
            'nearest': Image.NEAREST,
            'bilinear': Image.BILINEAR,
            'bicubic': Image.BICUBIC,
            'box': Image.BOX,
            'lanczos': Image.LANCZOS,
            'hamming': Image.HAMMING
        }

        def _pil_resize(src, size, resample):
            pil_img = Image.fromarray(src)
            pil_img = pil_img.resize(size, resample)
            return np.asarray(pil_img)

        if backend.lower() == "cv2":
            if isinstance(interpolation, str):
                interpolation = _cv2_interp_from_str[interpolation.lower()]
            # compatible with opencv < version 4.4.0
            elif interpolation is None:
                interpolation = cv2.INTER_LINEAR
            self.resize_func = partial(cv2.resize, interpolation=interpolation)
        elif backend.lower() == "pil":
            if isinstance(interpolation, str):
                interpolation = _pil_interp_from_str[interpolation.lower()]
            self.resize_func = partial(_pil_resize, resample=interpolation)
        else:
            logger.warning(
                f"The backend of Resize only support \"cv2\" or \"PIL\". \"f{backend}\" is unavailable. Use \"cv2\" instead."
            )
            self.resize_func = cv2.resize

    def __call__(self, src, size):
        return self.resize_func(src, size)
    
@TRANSFORMS.register()
class RandCropImage(object):
    """ random crop image
        https://github.com/PaddlePaddle/PaddleClas/blob/release/2.3/ppcls/data/preprocess/ops/operators.py
    """

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

        img_h, img_w = img.shape[:2]

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
   
@TRANSFORMS.register()
class ResizeImage(object):
    """ resize image
        https://github.com/PaddlePaddle/PaddleClas/blob/release/2.3/ppcls/data/preprocess/ops/operators.py
    """

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
        img_h, img_w = img.shape[:2]
        if self.resize_short is not None:
            percent = float(self.resize_short) / min(img_w, img_h)
            w = int(round(img_w * percent))
            h = int(round(img_h * percent))
        else:
            w = self.w
            h = self.h
        return self._resize_func(img, (w, h))


@TRANSFORMS.register()
class NormalizeImage(PT.Normalize):
    """NormalizeImage normalize input value to 0 ~ 1 in order to avoid overflow.
    ``output[channel] = (input[channel] / 255. - mean[channel]) / std[channel]``

    Args:
        scale (float): Normalize input value to [0, 1].
        mean (int|float|list|tuple): Sequence of means for each channel.
        std (int|float|list|tuple): Sequence of standard deviations for each channel.
        data_format (str, optional): Data format of img, should be 'HWC' or 
            'CHW'. Default: 'CHW'.
        to_rgb (bool, optional): Whether to convert to rgb. Default: False.
        keys (list[str]|tuple[str], optional): Same as ``BaseTransform``. Default: None.

    Shape:
        - img(PIL.Image|np.ndarray|Paddle.Tensor): The input image with shape (H x W x C).
        - output(PIL.Image|np.ndarray|Paddle.Tensor): A normalized array or tensor.

    Returns:
        A callable object of Normalize.

    Examples:
    
        .. code-block:: python

            import numpy as np
            from PIL import Image
            from paddle.vision.transforms import Normalize

            normalize = NormalizeImage(scale=1./255.,
                                  mean=[127.5, 127.5, 127.5], 
                                  std=[127.5, 127.5, 127.5],
                                  data_format='HWC')

            fake_img = Image.fromarray((np.random.rand(300, 320, 3) * 255.).astype(np.uint8))

            fake_img = normalize(fake_img)
            print(fake_img.shape)
            print(fake_img.max, fake_img.max)
    
    """

    def __init__(self,
                 scale=None,
                 mean=0.0,
                 std=1.0,
                 data_format='CHW',
                 to_rgb=False,
                 dtype='float32',
                 keys=None):
        super(NormalizeImage, self).__init__(mean=mean, std=std, keys=keys)
        self.scale = eval(scale)
        self.dtype = dtype 

    def _apply_image(self, img):
        if self.scale is not None:
            img = img * self.scale
        img = F.normalize(img, self.mean, self.std, self.data_format,
                           self.to_rgb)
        return img.astype(self.dtype)
