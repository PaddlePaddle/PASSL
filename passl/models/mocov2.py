# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

from collections.abc import Callable

import os
import copy
import numpy as np

import paddle
import paddle.nn as nn
from passl.nn import init
import paddle.nn.functional as F
from passl.models.base_model import Model
from paddle.nn.initializer import Constant, Normal
from functools import partial, reduce
from passl.models.resnet import ResNet
from paddle.vision.models.resnet import resnet50
import random
__all__ = [
    'mocov2_resnet50_linearprobe',
    'mocov2_resnet50_pretrain',
]

class MoCoV2Projector(nn.Layer):
    def __init__(self, with_pool, in_dim, out_dim):
        super().__init__()

        self.with_pool = with_pool
        if with_pool:
            self.avgpool = nn.Sequential(
                nn.AdaptiveAvgPool2D((1, 1)), nn.Flatten(start_axis=1))

        self.mlp = nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU())

    def forward(self, x):

        if self.with_pool:
            x = self.avgpool(x)

        x = self.mlp(x)
        return x


class MoCoClassifier(nn.Layer):
    def __init__(self, with_pool, num_features, class_num):
        super().__init__()

        self.with_pool = with_pool
        if with_pool:
            self.avgpool = nn.Sequential(
                nn.AdaptiveAvgPool2D((1, 1)), nn.Flatten(start_axis=1))

        self.fc = nn.Linear(num_features, class_num)
        normal_ = Normal(std=0.01)
        zeros_ = Constant(value=0.)

        normal_(self.fc.weight)
        zeros_(self.fc.bias)

    def save(self,path):
        paddle.save(self.fc.state_dict(),path + ".pdparams")
    def load(self,path):
        self.fc.set_state_dict(paddle.load(path+".pdparams"))
        

    def forward(self, x):

        if self.with_pool:
            x = self.avgpool(x)
        x = self.fc(x)
        return x


class MoCoV2Pretain(Model):
    """ MoCo v1, v2
    
    ref: https://github.com/facebookresearch/moco/blob/main/moco/builder.py
    ref: https://github.com/PaddlePaddle/PASSL/blob/main/passl/modeling/architectures/moco.py
    """

    def __init__(self,
                 base_encoder,
                 base_projector,
                 base_classifier,
                 momentum_encoder,
                 momentum_projector,
                 momentum_classifier,
                 dim=128,
                 K=65536,
                 m=0.999,
                 T=0.07,
                 **kwargs):
        super(MoCoV2Pretain, self).__init__()

        self.m = m
        self.T = T
        self.K = K

        self.base_encoder = nn.Sequential(base_encoder(), base_projector(),
                                          base_classifier())
        self.momentum_encoder = nn.Sequential(
            momentum_encoder(), momentum_projector(), momentum_classifier())

        for param_b, param_m in zip(self.base_encoder.parameters(),
                                    self.momentum_encoder.parameters()):
            param_m.copy_(param_b, False)  # initialize
            param_m.stop_gradient = True  # not update by gradient

        # create the queue
        self.register_buffer("queue", paddle.randn([dim, K]))
        self.queue = F.normalize(self.queue, axis=0)

        self.register_buffer("queue_ptr", paddle.zeros([1], 'int64'))

        self.loss_fuc = nn.CrossEntropyLoss()
    
    def save(self, path, local_rank=0, rank=0):
        paddle.save(self.state_dict(), path + ".pdparams")

        # rename moco pre-trained keys
        state_dict = self.state_dict()
        for k in list(state_dict.keys()):
            # retain only base_encoder up to before the embedding layer
            if k.startswith('base_encoder') and not k.startswith(
                    'base_encoder.head'):
                # remove prefix
                state_dict[k[len("base_encoder."):]] = state_dict[k]
            # delete renamed or unused k
            del state_dict[k]

        paddle.save(state_dict, path + "_base_encoder.pdparams")

    @paddle.no_grad()
    def _update_momentum_encoder(self):
        """Momentum update of the momentum encoder"""
        #Note(GuoxiaWang): disable auto cast when use mix_precision
        with paddle.amp.auto_cast(False):
            for param_b, param_m in zip(self.base_encoder.parameters(),
                                        self.momentum_encoder.parameters()):
                paddle.assign((param_m * self.m + param_b * (1. - self.m)),
                              param_m)
                param_m.stop_gradient = True

    # utils
    @paddle.no_grad()
    def concat_all_gather(self, tensor):
        """
        Performs all_gather operation on the provided tensors.
        """
        if paddle.distributed.get_world_size() < 2:
            return tensor
        tensors_gather = []
        paddle.distributed.all_gather(tensors_gather, tensor)

        output = paddle.concat(tensors_gather, axis=0)
        return output
    
    @paddle.no_grad()
    def _dequeue_and_enqueue(self, keys):
        keys = self.concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr[0])
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.transpose([1, 0])
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @paddle.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = self.concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = paddle.randperm(batch_size_all)

        # broadcast to all gpus
        if paddle.distributed.get_world_size() > 1:
            paddle.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = paddle.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = paddle.distributed.get_rank()
        idx_this = idx_shuffle.reshape([num_gpus, -1])[gpu_idx]
        return paddle.gather(x_gather, idx_this, axis=0), idx_unshuffle

    @paddle.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = self.concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = paddle.distributed.get_rank()
        idx_this = idx_unshuffle.reshape([num_gpus, -1])[gpu_idx]

        return paddle.gather(x_gather, idx_this, axis=0)
        
    def forward(self, inputs):
        assert isinstance(inputs, list)
        x1 = inputs[0]
        x2 = inputs[1]
        # compute query features
        q = self.base_encoder(x1)  # queries: NxC
        q = F.normalize(q, axis=1)

        # compute key features
        with paddle.no_grad():  # no gradient
            self._update_momentum_encoder()  # update the momentum encoder

            # shuffle for making use of BN
            k, idx_unshuffle = self._batch_shuffle_ddp(x2)

            k = self.momentum_encoder(k)  # keys: NxC
            k = F.normalize(k, axis=1)

            # undo shuffle
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = paddle.sum(q * k, axis=1).unsqueeze(-1)
        # negative logits: NxK
        l_neg = paddle.matmul(q, self.queue.clone().detach())

        # logits: Nx(1+K)
        logits = paddle.concat((l_pos, l_neg), axis=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = paddle.zeros([logits.shape[0]], dtype=paddle.int64)

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        return self.loss_fuc(logits, labels)

class MoCoV2LinearProbe(ResNet):
    """ MoCo v1, v2
    
    ref: https://github.com/facebookresearch/moco/blob/main/moco/builder.py
    ref: https://github.com/PaddlePaddle/PASSL/blob/main/passl/modeling/architectures/moco.py
    """

    def __init__(self,
                 **kwargs):
        super().__init__()
        # freeze all layers but the last fc
        for name, param in self.named_parameters():
            if name not in ['fc.weight', 'fc.bias']:
                param.stop_gradient = True

        # optimize only the linear classifier
        parameters = list(
            filter(lambda p: not p.stop_gradient, self.parameters()))
        assert len(parameters) == 2  # weight, bias

        init.normal_(self.fc.weight, mean=0.0, std=0.01)
        init.zeros_(self.fc.bias)
        self.apply(self._freeze_norm)

    def _freeze_norm(self, layer):
        if isinstance(layer, (nn.layer.norm._BatchNormBase)):
            layer._use_global_stats = True

    def load_pretrained(self, path, rank=0, finetune=False):
        if not os.path.exists(path + '.pdparams'):
            raise ValueError("Model pretrain path {} does not "
                             "exists.".format(path))

        path = path + ".pdparams"
        base_encoder_dict = paddle.load(path)
        for k in list(base_encoder_dict.keys()):
            # retain only encoder_q up to before the embedding layer
            if k.startswith('0.'):
                # remove prefix
                base_encoder_dict[k[len(
                    "0."):]] = base_encoder_dict[k]
                # delete renamed
                del base_encoder_dict[k]

        for name, param in self.state_dict().items():
            if name in base_encoder_dict and param.dtype != base_encoder_dict[
                    name].dtype:
                base_encoder_dict[name] = base_encoder_dict[name].cast(
                    param.dtype)

        self.set_state_dict(base_encoder_dict)

def mocov2_resnet50_linearprobe(**kwargs):
    # **kwargs specify numclass
    resnet = MoCoV2LinearProbe(with_pool=True,**kwargs)
    return resnet
def mocov2_resnet50_pretrain(**kwargs):
    # prepare all layer here
    base_encoder = partial(resnet50, with_pool=False,num_classes=0)
    base_projector = partial(MoCoV2Projector, with_pool=True, in_dim=2048,out_dim=2048)
    base_classifier = partial(MoCoClassifier, with_pool=False, num_features=2048, class_num=128)
    momentum_encoder = partial(resnet50, with_pool=False, num_classes=0)
    momentum_projector = partial(MoCoV2Projector,with_pool=True,in_dim=2048,out_dim=2048)
    momentum_classifier = partial(MoCoClassifier,with_pool=False,num_features=2048,class_num=128)
    model = MoCoV2Pretain(
        base_encoder=base_encoder,
        base_projector=base_projector,
        base_classifier=base_classifier,
        momentum_encoder=momentum_encoder,
        momentum_projector=momentum_projector,
        momentum_classifier=momentum_classifier,
        T=0.2,
        **kwargs)
    return model

if __name__ == "__main__":
    model = mocov2_resnet50_pretrain()
    model.save("./mocov2")
    model_lineprobe = mocov2_resnet50_linearprobe()
    model_lineprobe.load_pretrained("./mocov2_base_encoder")
