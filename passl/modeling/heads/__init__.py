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

from .contrastive_head import ContrastiveHead
from .clas_head import ClasHead
from .l2_head import L2Head
from .mb_head import MBHead
from .clip_head import CLIPHead
from .builder import build_head
from .simclr_contrastive_head import SimCLRContrastiveHead

