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

from ...modules.init import init_backbone_weight
from ...modules import freeze_batchnorm_statictis
from .builder import MODELS
from ..backbones import build_backbone
from ..necks import build_neck
from ..heads import build_head


@MODELS.register()
class MoCo(nn.Layer):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self,
                 backbone,
                 neck=None,
                 head=None,
                 dim=128,
                 K=65536,
                 m=0.999,
                 T=0.07):
        """
        Args:
            backbone (dict): config of backbone.
            neck (dict): config of neck.
            head (dict): config of head.
            scale (list|tuple): Range of size of the origin size cropped. Default: (0.08, 1.0)
            dim (int): feature dimension. Default: 128.
            K (int): queue size; number of negative keys. Default: 65536.
            m (float): moco momentum of updating key encoder. Default: 0.999.
            T (float): softmax temperature. Default: 0.07.
        """
        super(MoCo, self).__init__()

        self.K = K
        self.m = m
        self.T = T

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = nn.Sequential(build_backbone(backbone),
                                       build_neck(neck))
        self.encoder_k = nn.Sequential(build_backbone(backbone),
                                       build_neck(neck))

        self.backbone = self.encoder_q[0]

        self.head = build_head(head)

        for param_q, param_k in zip(self.encoder_q.parameters(),
                                    self.encoder_k.parameters()):
            param_k.set_value(param_q)  # initialize
            param_k.stop_gradient = True  # not update by gradient

        freeze_batchnorm_statictis(self.encoder_k)

        # create the queue
        self.register_buffer("queue", paddle.randn([dim, K]))
        self.queue = nn.functional.normalize(self.queue, axis=0)

        self.register_buffer("queue_ptr", paddle.zeros([1], 'int64'))

    @paddle.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(),
                                    self.encoder_k.parameters()):
            paddle.assign((param_k * self.m + param_q * (1. - self.m)), param_k)
            param_k.stop_gradient = True

    @paddle.no_grad()
    def _dequeue_and_enqueue(self, keys):
        keys = concat_all_gather(keys)

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
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = paddle.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        if paddle.distributed.get_world_size() > 1:
            paddle.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = paddle.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = paddle.distributed.get_rank()
        idx_this = idx_shuffle.reshape([num_gpus, -1])[gpu_idx]
        return paddle.index_select(x_gather, idx_this), idx_unshuffle

    @paddle.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = paddle.distributed.get_rank()
        idx_this = idx_unshuffle.reshape([num_gpus, -1])[gpu_idx]

        return paddle.index_select(x_gather, idx_this)

    def train_iter(self, *inputs, **kwargs):
        img_q, img_k = inputs

        # compute query features
        q = self.encoder_q(img_q)  # queries: NxC
        q = nn.functional.normalize(q, axis=1)

        # compute key features
        with paddle.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            im_k, idx_unshuffle = self._batch_shuffle_ddp(img_k)

            k = self.encoder_k(im_k)  # keys: NxC
            k = nn.functional.normalize(k, axis=1)

            # undo shuffle
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        # compute logits
        # FIXME: Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = paddle.sum(q * k, axis=1).unsqueeze(-1)
        # negative logits: NxK
        l_neg = paddle.matmul(q, self.queue.clone().detach())

        outputs = self.head(l_pos, l_neg)
        self._dequeue_and_enqueue(k)

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


@paddle.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    """
    if paddle.distributed.get_world_size() < 2:
        return tensor

    tensors_gather = []
    paddle.distributed.all_gather(tensors_gather, tensor)

    output = paddle.concat(tensors_gather, axis=0)
    return output 