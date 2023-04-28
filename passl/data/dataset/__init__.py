# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np
from PIL import Image
import cv2

_image_backend = "pil"


def set_image_backend(backend):
    """
    Specifies the package used to load images.

    Args:
        backend (string): Name of the image backend. one of {'PIL', 'accimage'}.
            The :mod:`accimage` package uses the Intel IPP library. It is
            generally faster than PIL, but does not support as many operations.
    """
    global _image_backend
    if backend not in ["pil", "cv2"]:
        raise ValueError(
            f"Invalid backend '{backend}'. Options are 'pil' and 'cv2'")
    _image_backend = backend


def get_image_backend():
    """
    Gets the name of the package used to load images
    """
    return _image_backend


def pil_loader(path: str):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        img = img.convert("RGB")
        return img


def cv2_loader(path: str):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)


def default_loader(path: str):
    if get_image_backend() == "cv2":
        return cv2_loader(path)
    else:
        return pil_loader(path)


from .imagenet_dataset import ImageNetDataset
from .imagefolder_dataset import ImageFolder
from .multicrop_dataset import MultiCropDataset
