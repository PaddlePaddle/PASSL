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

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from .builder import MODELS, build_model


def kldiv(x, target, eps=1e-12):
    class_num = x.shape[-1]
    cost = target * paddle.log(
        (target + eps) / (x + eps)) * class_num
    return cost

@MODELS.register()
class DistillationWrapper(nn.Layer):
    def __init__(self,
                 models=None,
                 pretrained_list=None,
                 freeze_params_list=None,
                 infer_model_key=None,
                 dml_loss_weight=0.5,
                 head_loss_weight=0.5):
        super().__init__()
        assert isinstance(models, list)
        self.model_dict = {}
        self.model_name_list = []
        if pretrained_list is not None:
            assert len(pretrained_list) == len(models)

        if freeze_params_list is None:
            freeze_params_list = [False] * len(models)
        assert len(freeze_params_list) == len(models)
        for idx, model_config in enumerate(models):
            assert len(model_config) == 1
            key = list(model_config.keys())[0]
            model_config = model_config[key]
            model = build_model(model_config)
            if freeze_params_list[idx]:
                for param in model.parameters():
                    param.trainable = False
            self.model_dict[key] = self.add_sublayer(key, model)
            self.model_name_list.append(key)
        if pretrained_list is not None:
            for idx, pretrained in enumerate(pretrained_list):
                if pretrained is not None:
                    state_dict = paddle.load(pretrained)
                    if 'state_dict' in state_dict:
                        state_dict = state_dict['state_dict']
                    self.model_dict[self.model_name_list[idx]]. \
                    set_state_dict(state_dict)
        self.infer_model_key = infer_model_key
        self.dml_loss_weight = dml_loss_weight
        self.head_loss_weight = head_loss_weight

    def train_iter(self, *inputs, **kwargs):
        img, label = inputs
        mixup_fn = kwargs['mixup_fn']
        if mixup_fn is not None:
            img, label = mixup_fn(img, label)

        outs = {}
        for key in self.model_name_list:
            x = self.model_dict[key].backbone_forward(img)
            outs[key] = self.model_dict[key].head(x)

        outs_act = [F.softmax(x, -1) for x in outs.values()]
        dml_loss = kldiv(outs_act[0], outs_act[1]) + \
                   kldiv(outs_act[1], outs_act[0])
        dml_loss = dml_loss.mean() / 2

        head_loss_input = (outs[self.infer_model_key], label)
        head_loss = self.model_dict[self.infer_model_key]. \
                         head.loss(*head_loss_input)

        losses = dict()
        losses['loss'] = self.dml_loss_weight * dml_loss + \
                         self.head_loss_weight * head_loss['loss']
        losses['acc1'], losses['acc5'] = head_loss['acc1'], head_loss['acc5']
        return losses

    def forward(self, *inputs, mode='train', **kwargs):
        if mode == 'train':
            return self.train_iter(*inputs, **kwargs)
        else:
            return self.model_dict[self.infer_model_key]. \
                        forward(self, *inputs, mode, **kwargs)
