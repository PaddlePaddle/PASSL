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

from .builder import build_model
from .byol_clas import ByolClassification
from .clas import Classification
from .moco import MoCo
from .simclr import SimCLR
from .simsiam import SimSiam
from .pixpro import PixPro
from .swav import SwAV

from .BEiTWrapper import BEiTWrapper, BEiTPTWrapper, BEiTFTWrapper
from .BYOL import BYOL
from .CaiTWrapper import CaiTWrapper
from .CLIPWrapper import CLIPWrapper
from .CvTWrapper import CvTWrapper
from .DeiTWrapper import DeiTWrapper
from .DistillationWrapper import DistillationWrapper
from .MAE import MAE_PRETRAIN, MAE_FINETUNE
from .MoCoBYOL import MoCoBYOL
from .MlpMixerWrapper import MlpMixerWrapper
from .SwinWrapper import SwinWrapper
from .T2TViTWrapper import T2TViTWrapper
from .ViTWrapper import ViTWrapper
from .LVViTWrapper import LVViTWrapper
