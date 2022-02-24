# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserve.
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
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import paddle.distributed as dist

from .builder import MODELS
from .builder import create_d_vae
from ..heads import build_head
from ..backbones import build_backbone


@MODELS.register()
class BEitWrapper(nn.Layer):
    def __init__(self, architecture=None, head=None):
        """A wrapper for a BEiT supervised model.

        Args:
            architecture (dict): A dictionary containing the BEiT instantiation parameters.
        """
        super().__init__()

        self.backbone = build_backbone(architecture)
        self.automatic_optimization = False
        self.head = build_head(head)

    def backbone_forward(self, x):
        x = self.backbone(x)
        return x

    def train_iter(self, *inputs, **kwargs):
        img, label = inputs
        cls_token = self.backbone_forward(img)
        outs = self.head(cls_token)
        loss_inputs = (outs, label)
        outputs = self.head.loss(*loss_inputs)
        return outputs

    def forward(self, *inputs, mode='train', **kwargs):
        if mode == 'train':
            return self.train_iter(*inputs, **kwargs)
        elif mode == 'test':
            return self.test_iter(*inputs, **kwargs)
        elif mode == 'extract':
            return self.backbone(*inputs)
        else:
            raise Exception("No such mode: {}".format(mode))

    def validation_step(self, val_batch, idx):
        image, text = val_batch
        image_logits, text_logits = self.forward(image, text)
        ground_truth = paddle.arange(len(image_logits))
        loss = (self.image_loss(image_logits, ground_truth) +
                self.text_loss(text_logits, ground_truth)).div(2)
        self.log('val_loss', loss)


@MODELS.register()
class BEiTPTWrapper(nn.Layer):
    def __init__(self, architecture=None, head=None, d_vae=None):
        """A wrapper for a BEiT Pretrain.

        Args:
            architecture (dict): A dictionary containing the BEiT instantiation parameters.
        """
        super().__init__()

        self.backbone = build_backbone(architecture)
        self.automatic_optimization = False
        self.head = build_head(head)
        with paddle.no_grad():
            self.d_vae = create_d_vae(d_vae)

    def get_codebook_indices(self, images):
        with paddle.no_grad():
            logits = self.d_vae.encoder(images)
            codebook_indices = logits.argmax(axis=1)
            return codebook_indices

    def backbone_forward(self,
                         x,
                         bool_masked_pos=None,
                         return_all_tokens=False):
        x = self.backbone(x,
                          bool_masked_pos=bool_masked_pos,
                          return_all_tokens=return_all_tokens)
        return x

    def train_iter(self, *inputs, **kwargs):
        samples, images, bool_masked_pos = inputs

        with paddle.no_grad():
            input_ids = self.get_codebook_indices(images).flatten(1)
            bool_masked_pos = bool_masked_pos.flatten(1).astype(
                'bool')  # to bool.
            labels = input_ids[bool_masked_pos]

        outputs = self.backbone_forward(samples,
                                        bool_masked_pos=bool_masked_pos,
                                        return_all_tokens=False)
        loss = self.head(outputs, labels)
        return loss

    def test_iter(self, *inputs, **kwargs):
        with paddle.no_grad():
            img, label = inputs
            x = self.backbone_forward(img)
            outs = self.head(x)

        return outs

    def forward(self, *inputs, mode='train', **kwargs):
        if mode == 'train':
            return self.train_iter(*inputs, **kwargs)
        elif mode == 'test':
            return self.test_iter(*inputs, **kwargs)
        elif mode == 'extract':
            return self.backbone(*inputs)
        else:
            raise Exception("No such mode: {}".format(mode))

    def validation_step(self, val_batch, idx):
        image, text = val_batch
        image_logits, text_logits = self.forward(image, text)
        ground_truth = paddle.arange(len(image_logits))
        loss = (self.image_loss(image_logits, ground_truth) +
                self.text_loss(text_logits, ground_truth)).div(2)
        self.log('val_loss', loss)


@MODELS.register()
class BEiTFTWrapper(nn.Layer):
    def __init__(self, architecture=None, head=None):
        """A wrapper for a BEiT Finetune.

        Args:
            architecture (dict): A dictionary containing the BEiT instantiation parameters.
        """
        super().__init__()
        self.backbone = build_backbone(architecture)
        self.head = build_head(head)

    def backbone_forward(self, x):
        x = self.backbone(x)
        return x

    def train_iter(self, *inputs, **kwargs):
        img, label = inputs
        mixup_fn = kwargs['mixup_fn']
        if mixup_fn is not None:
            img, label = mixup_fn(img, label)

        # Only Used For Debug The Network.
        import numpy as np
        img = paddle.to_tensor(np.load('ft_data.npy'))
        label = paddle.to_tensor(np.load('ft_tar.npy'))

        x = self.backbone_forward(img)
        outputs = self.head(x)
        outputs = self.head.loss(outputs, label)
        return outputs

    def test_iter(self, *inputs, **kwargs):
        with paddle.no_grad():
            img, _ = inputs
            x = self.backbone_forward(img)
            outs = self.head(x)
            return outs  # self.head.loss(outs, label, soft=False)

    def forward(self, *inputs, mode='train', **kwargs):
        if mode == 'train':
            return self.train_iter(*inputs, **kwargs)
        elif mode == 'test':
            return self.test_iter(*inputs, **kwargs)
        elif mode == 'extract':
            return self.backbone(*inputs)
        else:
            raise Exception("No such mode: {}".format(mode))
