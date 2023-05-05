# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import os
import numpy as np
from sys import flags
from collections import defaultdict

import paddle
import paddle.nn as nn

from passl.nn import init
from passl.scheduler import build_lr_scheduler
from passl.utils import logger
from passl.models.swav_resnet import swavresnet50
from passl.models.base_model import Model


__all__ = [
    'swav_resnet50_finetune',
    'swav_resnet50_linearprobe',
    'swav_resnet50_pretrain',
    'SwAV',
    'SwAVLinearProbe',
    'SwAVFinetune',
    'SwAVPretrain',
]

class SwAV(Model):
    def __init__(self, **kwargs):
        super().__init__()
        self.res_model = swavresnet50(**kwargs)

    def _load_model(self, path, model, tag):
        path = path + ".pdparams"
        if os.path.isfile(path):
            para_state_dict = paddle.load(path)

            # resnet
            model_state_dict = model.state_dict()
            keys = model_state_dict.keys()
            num_params_loaded = 0
            for k in keys:
                if k not in para_state_dict:
                    print("{} is not in pretrained model".format(k))
                elif list(para_state_dict[k].shape) != list(model_state_dict[k]
                                                            .shape):
                    print(
                        "[SKIP] Shape of pretrained params {} doesn't match.(Pretrained: {}, Actual: {})"
                        .format(k, para_state_dict[k].shape, model_state_dict[k]
                                .shape))
                else:
                    # conpact FP16 saving pretrained weight
                    if model_state_dict[k].dtype != para_state_dict[k].dtype:
                        para_state_dict[k] = para_state_dict[k].astype(model_state_dict[k].dtype)
                    model_state_dict[k] = para_state_dict[k]
                    num_params_loaded += 1
            model.set_dict(model_state_dict)
            print("There are {}/{} variables loaded into {}.".format(
                num_params_loaded, len(model_state_dict), tag))
        else:
            print("No pretrained weights found in {} => training with random weights".format(tag))

    def load_pretrained(self, path, rank=0, finetune=False):
        pass

    def save(self, path, local_rank=0, rank=0):
        paddle.save(self.state_dict(), path + ".pdparams")

    def _freeze_norm(self, layer):
        if isinstance(layer, (nn.layer.norm._BatchNormBase)):
            layer._use_global_stats = True

class SwAVLinearProbe(SwAV):
    def __init__(self, class_num=1000, **kwargs):
        super().__init__(**kwargs)
        self.linear = RegLog(class_num)
        self.res_model.eval()

        # freeze all layers but the last fc
        for name, param in self.named_parameters():
            if name not in ['linear.linear.weight', 'linear.linear.bias']:
                param.stop_gradient = True

        # optimize only the linear classifier
        parameters = list(
            filter(lambda p: not p.stop_gradient, self.parameters()))
        assert len(parameters) == 2  # weight, bias

        self.apply(self._freeze_norm)

    def load_pretrained(self, path, rank=0, finetune=False):
        self._load_model(path, self.res_model, 'backbone')

    def forward(self, inp):
        with paddle.no_grad():
            output = self.res_model(inp)
        output = self.linear(output)

        return output

class SwAVFinetune(SwAV):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.apply(self._freeze_norm)

    def load_pretrained(self, path, rank=0, finetune=False):
        self._load_model(path, self.res_model, 'backbone')

    def forward(self, inp):
        return self.res_model(inp)

class SwAVPretrain(SwAV):
    def __init__(self, queue_length=0, crops_for_assign=(0, 1), nmb_crops=[2, 6], epsilon=0.05, freeze_prototypes_niters=5005, **kwargs):
        super().__init__(**kwargs)
        self.crops_for_assign = crops_for_assign
        self.nmb_crops = nmb_crops
        self.temperature = 0.1
        self.epsilon = epsilon
        self.freeze_prototypes_niters = freeze_prototypes_niters

        self.apply(self._freeze_norm)

    def load_pretrained(self, path, rank=0, finetune=False):
        self._load_model('swav_800ep_pretrain.pdparams', self.res_model, 'backbone')

    @paddle.no_grad()
    def distributed_sinkhorn(self, out, sinkhorn_iterations=3):
        Q = paddle.exp(x=out / self.epsilon).t()
        B = Q.shape[1] * 4
        K = Q.shape[0]
        sum_Q = paddle.sum(x=Q)
        paddle.distributed.all_reduce(sum_Q)
        Q /= sum_Q
        for it in range(sinkhorn_iterations):
            sum_of_rows = paddle.sum(x=Q, axis=1, keepdim=True)
            paddle.distributed.all_reduce(sum_of_rows)
            Q /= sum_of_rows
            Q /= K
            Q /= paddle.sum(x=Q, axis=0, keepdim=True)
            Q /= B
        Q *= B
        return Q.t()

    def forward(self, inp):
        bs = inp[0].shape[0]

        # normalize the prototypes
        with paddle.no_grad():
            w = self.res_model.prototypes.weight.clone()
            w = paddle.nn.functional.normalize(x=w, axis=0, p=2) # 1
            paddle.assign(w, self.res_model.prototypes.weight)
        embedding, output = self.res_model(inp)
        embedding = embedding.detach()

        # compute loss
        loss = 0
        for i, crop_id in enumerate(self.crops_for_assign):
            with paddle.no_grad():
                out = output[bs * crop_id:bs * (crop_id + 1)].detach()
                q = self.distributed_sinkhorn(out)[-bs:]

            subloss = 0
            for v in np.delete(np.arange(np.sum(self.nmb_crops)), crop_id):
                x = output[bs * v:bs * (v + 1)] / self.temperature
                subloss -= paddle.mean(x=paddle.sum(x=q * paddle.nn.
                    functional.log_softmax(x=x, axis=1), axis=1))

            loss += subloss / (np.sum(self.nmb_crops) - 1)
        loss /= len(self.crops_for_assign)

        return loss

    def after_loss_backward(self, iteration):
        if iteration < self.freeze_prototypes_niters:
            for name, p in self.res_model.named_parameters():
                if 'prototypes' in name and p.grad is not None:
                    p.clear_grad()

def swav_resnet50_linearprobe(**kwargs):
    model = SwAVLinearProbe(**kwargs)
    return model

def swav_resnet50_finetune(**kwargs):
    model = SwAVFinetune(**kwargs)
    if paddle.distributed.get_world_size() > 1:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    return model

def swav_resnet50_pretrain(apex, **kwargs): # todo
    flags = {}
    flags['FLAGS_cudnn_exhaustive_search'] = True
    flags['FLAGS_cudnn_deterministic'] = False
    paddle.set_flags(flags)

    model = SwAVPretrain(**kwargs)

    if paddle.distributed.get_world_size() > 1:
        if not apex:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        else:
            # with apex syncbn speeds up computation than global syncbn
            process_group = apex.parallel.create_syncbn_process_group(8)
            model = apex.parallel.convert_syncbn_model(model, process_group=process_group)

    return model

class RegLog(paddle.nn.Layer):
    """Creates logistic regression on top of frozen features"""

    def __init__(self, num_labels):
        super(RegLog, self).__init__()
        s = 2048
        self.av_pool = paddle.nn.AdaptiveAvgPool2D(output_size=(1, 1))
        self.linear = paddle.nn.Linear(in_features=s, out_features=num_labels)

        init.normal_(self.linear.weight, mean=0.0, std=0.01)
        init.zeros_(self.linear.bias)

    def forward(self, x):
        x = self.av_pool(x)
        x = x.reshape((x.shape[0], -1))
        return self.linear(x)
