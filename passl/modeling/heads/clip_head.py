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

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from .builder import HEADS


@HEADS.register()
class CLIPHead(nn.Layer):
    def __init__(self):
        super(CLIPHead, self).__init__()
        self.image_loss = nn.CrossEntropyLoss()
        self.text_loss = nn.CrossEntropyLoss()
    
    def forward(self, image, text, image_mbs, text_mbs, ims, txt, batch_offset):
        outputs = dict()
        outputs['image'] = image
        outputs['text'] = text
        outputs['image_mbs'] = image_mbs 
        outputs['text_mbs'] = text_mbs
        outputs['ims'] = ims
        outputs['txt'] = txt
        outputs['batch_offset'] = batch_offset
        return outputs
