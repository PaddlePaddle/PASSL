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
import functools
import numpy as np
from sys import flags

import paddle
import paddle.nn as nn

from passl.nn import init
from passl.utils import logger
from passl.utils.infohub import runtime_info_hub
from passl.models.base_model import Model
from passl.models.resnet import ResNet, BottleneckBlock

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
        backbone_config = kwargs['backbone']
        backbone_type = backbone_config.pop("type", None)
        if backbone_type is not None:
            self.res_model = eval(backbone_type)(**backbone_config)
        else:
            AttributeError(f'Backbone type is not assigned, please assign it.')

    def _load_model(self, path, tag):
        path = path + ".pdparams"
        if os.path.isfile(path):
            para_state_dict = paddle.load(path)

            # resnet
            model_state_dict = self.state_dict()
            keys = model_state_dict.keys()
            num_params_loaded = 0
            for k in keys:
                if k not in para_state_dict:
                    logger.info("{} is not in pretrained model".format(k))
                elif list(para_state_dict[k].shape) != list(model_state_dict[k]
                                                            .shape):
                    logger.info(
                        "[SKIP] Shape of pretrained params {} doesn't match.(Pretrained: {}, Actual: {})"
                        .format(k, para_state_dict[k].shape, model_state_dict[k]
                                .shape))
                else:
                    # conpact FP16 saving pretrained weight
                    if model_state_dict[k].dtype != para_state_dict[k].dtype:
                        para_state_dict[k] = para_state_dict[k].astype(model_state_dict[k].dtype)
                    model_state_dict[k] = para_state_dict[k]
                    num_params_loaded += 1
            self.set_dict(model_state_dict)
            logger.info("There are {}/{} variables loaded into {} with {}.".format(
                num_params_loaded, len(model_state_dict), tag, path))
        else:
            logger.info("No pretrained weights found in {} => training with random weights".format(tag))

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
        self.linear = RegLogit(class_num)
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
        self._load_model(path, 'backbone')

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
        self._load_model(path, 'backbone')

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

        if runtime_info_hub.total_iterations < self.freeze_prototypes_niters:
            for name, p in self.res_model.named_parameters():
                if 'prototypes' in name:
                    p.stop_gradient = True
        else:
            for name, p in self.res_model.named_parameters():
                if 'prototypes' in name:
                    p.stop_gradient = False

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



class RegLogit(paddle.nn.Layer):
    """Creates logistic regression on top of frozen features"""

    def __init__(self, num_labels):
        super(RegLogit, self).__init__()
        s = 2048
        self.av_pool = paddle.nn.AdaptiveAvgPool2D(output_size=(1, 1))
        self.linear = paddle.nn.Linear(in_features=s, out_features=num_labels)

        init.normal_(self.linear.weight, mean=0.0, std=0.01)
        init.zeros_(self.linear.bias)

    def forward(self, x):
        x = self.av_pool(x)
        x = x.reshape((x.shape[0], -1))
        return self.linear(x)


def kaiming_normal_init(param, **kwargs):
    initializer = nn.initializer.KaimingNormal(**kwargs)
    initializer(param, param.block)

def constant_init(param, **kwargs):
    initializer = nn.initializer.Constant(**kwargs)
    initializer(param, param.block)


class SwAVResNet(paddle.nn.Layer):
    def __init__(self, block, depth,
        normalize=False, output_dim=0, hidden_mlp=0,
        nmb_prototypes=0, eval_mode=False):

        super(SwAVResNet, self).__init__()
        self.l2norm = normalize
        self.eval_mode = eval_mode
        num_out_filters = 512

        self.avgpool = paddle.nn.AdaptiveAvgPool2D(output_size=(1, 1))

        if output_dim == 0:
            self.projection_head = None
        elif hidden_mlp == 0:
            self.projection_head = paddle.nn.Linear(in_features=
                num_out_filters * block.expansion, out_features=output_dim)
        else:
            self.projection_head = paddle.nn.Sequential(paddle.nn.Linear(
                in_features=num_out_filters * block.expansion, out_features
                =hidden_mlp), paddle.nn.BatchNorm1D(num_features=hidden_mlp,
                momentum=1 - 0.1, epsilon=1e-05, weight_attr=None,
                bias_attr=None, use_global_stats=True), paddle.nn.ReLU(),
                paddle.nn.Linear(in_features=hidden_mlp, out_features=
                output_dim))

        self.prototypes = None
        if isinstance(nmb_prototypes, list):
            self.prototypes = MultiPrototypes(output_dim, nmb_prototypes)
        elif nmb_prototypes > 0:
            self.prototypes = paddle.nn.Linear(in_features=output_dim,
                out_features=nmb_prototypes, bias_attr=False)
            for sublayer in self.sublayers():
                if isinstance(sublayer, nn.Conv2D):
                    kaiming_normal_init(sublayer.weight)
                elif isinstance(sublayer, (nn.BatchNorm2D, nn.GroupNorm)):
                    constant_init(sublayer.weight, value=1.0)
                    constant_init(sublayer.bias, value=0.0)

        self.encoder = functools.partial(ResNet, block=block, depth=depth)(with_pool=False, class_num=0)

    def forward_backbone(self, x):
        x = self.encoder(x)

        if self.eval_mode:
            return x

        x = self.avgpool(x)
        x = paddle.flatten(x=x, start_axis=1)
        return x

    def forward_head(self, x):
        if self.projection_head is not None:
            x = self.projection_head(x)
        if self.l2norm:
            x = paddle.nn.functional.normalize(x=x, axis=1, p=2)
        if self.prototypes is not None:
            return x, self.prototypes(x)
        return x

    def forward(self, inputs):
        if not isinstance(inputs, list):
            inputs = [inputs]

        idx_crops = paddle.cumsum(x=paddle.unique_consecutive(x=paddle.
            to_tensor(data=[inp.shape[-1] for inp in inputs]),
            return_counts=True)[1], axis=0) # padiff
        start_idx = 0
        for end_idx in idx_crops:
            _out = self.forward_backbone(paddle.concat(x=inputs[start_idx:end_idx]))
            if start_idx == 0:
                output = _out
            else:
                output = paddle.concat(x=(output, _out))
            start_idx = end_idx
        return self.forward_head(output)


class MultiPrototypes(paddle.nn.Layer):
    def __init__(self, output_dim, nmb_prototypes):
        super(MultiPrototypes, self).__init__()
        self.nmb_heads = len(nmb_prototypes)
        for i, k in enumerate(nmb_prototypes):
            self.add_module('prototypes' + str(i), paddle.nn.Linear(
                in_features=output_dim, out_features=k, bias_attr=False))

    def forward(self, x):
        out = []
        for i in range(self.nmb_heads):
            out.append(getattr(self, 'prototypes' + str(i))(x))
        return out


def swavresnet50(**kwargs):
    return SwAVResNet(block=BottleneckBlock, depth=50, **kwargs)


def swav_resnet50_linearprobe(**kwargs):
    model = SwAVLinearProbe(**kwargs)
    return model

def swav_resnet50_finetune(**kwargs):
    model = SwAVFinetune(**kwargs)
    if paddle.distributed.get_world_size() > 1:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    return model

def swav_resnet50_pretrain(apex, **kwargs):
    model = SwAVPretrain(**kwargs)

    if paddle.distributed.get_world_size() > 1:
        if not apex:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        else:
            # with apex syncbn speeds up computation than global syncbn
            process_group = apex.parallel.create_syncbn_process_group(8)
            model = apex.parallel.convert_syncbn_model(model, process_group=process_group)

    return model
