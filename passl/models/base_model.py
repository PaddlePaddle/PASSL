# copyright (c) 2022 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import ABCMeta, abstractmethod

import paddle
import paddle.nn as nn


class Model(nn.Layer):
    __metaclass__ = ABCMeta

    def __init__(self):
        super().__init__()

    @abstractmethod
    def load_pretrained(self, path, rank=0, finetune=False):
        raise Exception(
            "NotImplementedError, you must overwrite load_pretrained method in subclass."
        )

    @abstractmethod
    def save(self, path, local_rank=0, rank=0):
        raise Exception(
            "NotImplementedError, you must overwrite save method in subclass.")
