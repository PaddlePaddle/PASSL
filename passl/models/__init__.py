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

import copy
import importlib

from .base_model import Model
from .resnet import *
from .vision_transformer import *
from .vision_transformer_hybrid import *
from .deit import *
from .cait import *
from .mae import *
from .convmae import *
from .swin_transformer import *
from .cae import *
from .convnext import *
from .mocov3 import *
from .swav import *
from .simsiam import *
from .dino import *
from .dinov2 import *

__all__ = ["build_model"]


def build_model(config):
    config = copy.deepcopy(config)
    model_type = config.pop("name")
    mod = importlib.import_module(__name__)
    model = getattr(mod, model_type)(**config)
    assert isinstance(model,
                      Model), 'model must inherit from passl.models.Model'
    return model
