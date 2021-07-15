# copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
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

"""Contains image preprocessing methods."""
import cv2
import math
import random
import numpy as np


def to_rgb_bgr(img):
    """ convert BRG->RGB, or RGB->BGR
    Args:
        img: HWC(bgr or rgb)
    Returns:
        converted image.
    """
    return img[:, :, ::-1]


def to_chw(img):
    """ HWC to CHW """
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img = img.transpose((2, 0, 1))
    return img


def normalize(img, mean=None, std=None):
    """ Normalize img with HWC(rgb) in the range[0, 255]
        img = img / 255.0
        img = (img - mean) / std
    Args:
        img: HWC(rgb) in the range [0, 255]
        mean: means for each channel, if None,
        std:
    Returns:
        normalize img HWC(rgb).
    """
    # to HWC(rgb) in the range[0.0, 1.0]
    img = img / np.float32(255.0)

    if mean is None or std is None:
        return img

    return (img - mean) / std


def hflip(img):
    # return cv2.flip(img, 1)
    return img[:, ::-1, :]


def vflip(img):
    # return cv2.flip(img, 0)
    return img[::-1, :, :]


def adjust_brightness(img, factor):
    # from https://github.com/victorca25/opencv_transforms/blob/master/opencv_transforms/functional.py
    table = np.array([i*factor for i in range(0, 256)]).clip(0, 255).astype('uint8')
    if img.shape[2] == 1:
        return cv2.LUT(img, table)[:, :, np.newaxis]
    else:
        return cv2.LUT(img, table)


def adjust_contrast(img, factor):
    # alt 1:
    # table = np.array([(i-74) * factor + 74 for i in range(0, 256)]).clip(0, 255).astype('uint8')
    # if img.shape[2] == 1:
    #     return cv2.LUT(img, table)[:, :, np.newaxis]
    # else:
    #     return cv2.LUT(img, table)

    # alt 2:
    # same results
    im = img.astype(np.float32)
    mean = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY).mean()
    im = (1.0 - factor) * mean + factor * im
    im = im.clip(min=0, max=255)
    return im.astype(img.dtype)


def adjust_saturation(img, factor):
    degenerate = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)
    im = np.float32(1.0 - factor) * degenerate + np.float32(factor) * img
    im = im.clip(min=0, max=255)
    return im.astype(img.dtype)

# def adjust_saturation(img, factor):
#     gray = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
#     gray = np.expand_dims(gray, axis=-1)
#     im = img * factor + gray * (1.0 - factor)
#     im = im.clip(min=0, max=255)
#     return im.astype(img.dtype)


def adjust_hue(img, factor):
    # https://github.com/victorca25/opencv_transforms/blob/master/opencv_transforms/functional.py
    # im = img.astype(np.uint8)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV_FULL)

    hsv[..., 0] += np.uint8(factor * 255)

    im = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB_FULL)
    return im.astype(img.dtype)


def to_grayscale(img):
    img = np.broadcast_to(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)[:, :, np.newaxis], img.shape)
    return img


def resize(img, size, interpolation=cv2.INTER_LINEAR):
    """ resize """
    if isinstance(size, int):
        h, w = img.shape[:2]  # hwc

        min_hw = min(h, w)
        if size == min_hw:
            return img

        percent = float(size) / min_hw
        rw = int(round(w * percent))
        rh = int(round(h * percent))
        out = cv2.resize(img, (rw, rh), interpolation=interpolation)
    else:
        # size (h, w)
        out = cv2.resize(img, (size[1], size[0]), interpolation=interpolation)

    return out


def crop(img, h_start, w_start, h, w):
    h_end = h_start + h
    w_end = w_start + w
    return img[h_start:h_end, w_start:w_end, :]


def center_crop(img, size):
    if type(size) is int:
        size = (size, size)

    h, w = size
    img_h, img_w = img.shape[:2]
    w_start = int(round(img_w - w) / 2.)
    h_start = int(round(img_h - h) / 2.)

    return crop(img, h_start, w_start, h, w)


def random_crop_with_resize(img,
                            size,
                            scale=(0.08, 1.0),
                            ratio=(3./4., 4./3.),
                            interpolation=cv2.INTER_LINEAR):
    """ random_crop and resized to given size"""
    if type(size) is int:
        size = (size, size)
    aspect_ratio = math.sqrt(random.uniform(*ratio))
    w = 1. * aspect_ratio
    h = 1. / aspect_ratio

    bound = min((float(img.shape[1]) / img.shape[0]) / (w ** 2),
                (float(img.shape[0]) / img.shape[1]) / (h ** 2))
    scale_max = min(scale[1], bound)
    scale_min = min(scale[0], bound)

    target_area = img.shape[0] * img.shape[1] * random.uniform(scale_min, scale_max)
    target_size = math.sqrt(target_area)
    w = int(target_size * w)
    h = int(target_size * h)

    i = random.randint(0, img.shape[0] - h)
    j = random.randint(0, img.shape[1] - w)

    img = img[i:i+h, j:j+w, :]
    resized = cv2.resize(img, dsize=(size[1], size[0]),
                         interpolation=interpolation)
    return resized


# ====================================


def rotate_image(img):
    """ rotate_image """
    (h, w) = img.shape[:2]
    center = (w / 2, h / 2)
    angle = random.randint(-10, 10)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h))
    return rotated


def crop_image(img, size, center):
    """ crop_image """
    height, width = img.shape[:2]
    if center == True:
        w_start = (width - size) / 2
        h_start = (height - size) / 2
    else:
        w_start = random.randint(0, width - size)
        h_start = random.randint(0, height - size)
    w_end = w_start + size
    h_end = h_start + size
    img = img[h_start:h_end, w_start:w_end, :]
    return img

