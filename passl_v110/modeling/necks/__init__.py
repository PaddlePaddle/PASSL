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

from .base_neck import LinearNeck
from .base_neck import DenseCLNeck
from .base_neck import NonLinearNeckV1
from .base_neck import NonLinearNeckV2
from .base_neck import NonLinearNeckV3

from .base_neck import NonLinearNeckfc3
from .base_neck import NonLinearNeckfc3V2
from .base_neck import SwAVNeck
from .base_neck import DINONeck
from .builder import build_neck
