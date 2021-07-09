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
from paddle.nn import functional as F
import paddle.distributed as dist
import copy
import numpy as np

from .hook import Hook
from .builder import HOOKS

@HOOKS.register()
class CLIPOptimizerHook(Hook):
    def __init__(self, priority=1):
        self.priority = priority

    def get_image_logits(self, x, text_embed):
        return paddle.matmul(F.normalize(
            self.model.encode_image(x), axis=1), text_embed.t()) * self.model.logit_scale.exp()

    def get_text_logits(self, x, image_embed):
        return paddle.matmul(F.normalize(
            self.model.encode_text(x), axis=1), image_embed.t()) * self.model.logit_scale.exp()
    

    def train_iter_end(self, trainer):
        outputs = trainer.outputs
        image = outputs['image']
        text = outputs['text']
        image_mbs = outputs['image_mbs']
        text_mbs = outputs['text_mbs']
        ims = outputs['ims']
        txt = outputs['txt']
        batch_offset = outputs['batch_offset']
        minibatch_size = trainer.model._layers.minibatch_size
        with paddle.no_grad():
            image_logits = paddle.matmul(
                paddle.concat(ims), paddle.concat(txt).t()) * trainer.model._layers.model.logit_scale.exp().reshape((1,))
            ground_truth = paddle.arange(len(image_logits)).astype('int64')
            loss = (F.cross_entropy(
                image_logits, ground_truth) + F.cross_entropy(image_logits.t(), ground_truth)) / 2
            acc_i = (paddle.argmax(image_logits, 1) == ground_truth).astype('int64').sum()
            acc_t = (paddle.argmax(image_logits, 0) == ground_truth).astype('int64').sum()
            #self.log_dict({'loss': loss, 'acc': (acc_i + acc_t) / 2 / len(image)}, prog_bar=True)

        for i_opt in range(len(trainer.optimizer)):
            if 'lars' in trainer.optimizer[i_opt].type:
                trainer.optimizer[i_opt].clear_gradients()
            else:
                trainer.optimizer[i_opt].clear_grad()

        # image loss
        for j, mb in enumerate(image_mbs):
            images_tmp = copy.deepcopy(ims)
            images_tmp[dist.get_rank()][j*minibatch_size:(j+1)*minibatch_size] = F.normalize(trainer.model._layers.model.encode_image(mb), axis=1)
            image_logits = paddle.matmul(paddle.concat(images_tmp), paddle.concat(txt).t()) * trainer.model._layers.model.logit_scale.exp().reshape((1,))
            ground_truth = paddle.arange(len(image_logits)).astype('int64')
            loss = (F.cross_entropy(image_logits, ground_truth) + F.cross_entropy(image_logits.t(), ground_truth)) / 2
            loss.backward()

        # text loss
        for j, mb in enumerate(text_mbs):
            text_tmp = copy.deepcopy(txt)
            text_tmp[dist.get_rank()][j*minibatch_size:(j+1)*minibatch_size] = F.normalize(trainer.model._layers.model.encode_text(mb), axis=1)
            image_logits = paddle.matmul(paddle.concat(ims), paddle.concat(text_tmp).t()) * trainer.model._layers.model.logit_scale.exp().reshape((1,))
            loss = (F.cross_entropy(image_logits, ground_truth) + F.cross_entropy(image_logits.t(), ground_truth))/2
            loss.backward()

        for i_opt in range(len(trainer.optimizer)):
            if 'lars' in trainer.optimizer[0].type:
                trainer.optimizer[i_opt].minimize(loss)
            else:
                trainer.optimizer[i_opt].step()
            trainer.model._layers.model.logit_scale.clip(-float('inf'), np.log(100))

        trainer.outputs = {}
        trainer.outputs['loss'] = loss
        trainer.outputs['acc'] = (acc_i + acc_t) / 2 / len(image)