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

from ...modules import freeze_batchnorm_statictis
from .builder import MODELS
from ..backbones import build_backbone
from ..necks import build_neck
from ..heads import build_head


@MODELS.register()
class DenseCL(nn.Layer):
    """
    Build a DenseCL model with: a query encoder, a key encoder, and a queue.
    https://arxiv.org/abs/2011.09157.
    """
    def __init__(self,
                 backbone,
                 neck=None,
                 head=None,
                 dim=128,
                 K=65536,
                 m=0.999,
                 T=0.07,
                 loss_lambda=0.5):
        """
        Args:
            backbone (dict): config of backbone.
            neck (dict): config of neck.
            head (dict): config of head.
            dim (int): feature dimension. Default: 128.
            K (int): queue size; number of negative keys. Default: 65536.
            m (float): moco momentum of updating key encoder. Default: 0.999.
            T (float): softmax temperature. Default: 0.07.
            loss_lambda (float): coefficients to balance the local and global loss. Default: 0.5.
        """
        super(DenseCL, self).__init__()

        self.K = K
        self.m = m
        self.T = T
        self.loss_lambda = loss_lambda

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
        # create the second queue for dense output
        self.register_buffer("queue2", paddle.randn([dim, K]))
        self.queue2 = nn.functional.normalize(self.queue2, axis=0)
        self.register_buffer("queue2_ptr", paddle.zeros([1], 'int64'))

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
        """Update queue."""
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr[0])
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.transpose([1, 0])
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @paddle.no_grad()
    def _dequeue_and_enqueue2(self, keys):
        """Update queue2."""
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue2_ptr[0])
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue2[:, ptr:ptr + batch_size] = keys.transpose([1, 0])
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue2_ptr[0] = ptr

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
        q_b = self.encoder_q[0](img_q)  # backbone features
        q, q_grid, _ = self.encoder_q[1](q_b)  # queries: NxC, NxCxS^2
        q_b = q_b.reshape([q_b.shape[0], q_b.shape[1], -1])

        q = nn.functional.normalize(q, axis=1)
        q_b = nn.functional.normalize(q_b, axis=1)
        q_grid = nn.functional.normalize(q_grid, axis=1)

        # compute key features
        with paddle.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            im_k, idx_unshuffle = self._batch_shuffle_ddp(img_k)

            k_b = self.encoder_k[0](im_k)
            k, k_grid, k2 = self.encoder_k[1](k_b)  # keys: NxC, NxCxS^2
            k_b = k_b.reshape([k_b.shape[0], k_b.shape[1], -1])

            k = nn.functional.normalize(k, axis=1)
            k2 = nn.functional.normalize(k2, axis=1)
            k_b = nn.functional.normalize(k_b, axis=1)
            k_grid = nn.functional.normalize(k_grid, axis=1)

            # undo shuffle
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)
            k2 = self._batch_unshuffle_ddp(k2, idx_unshuffle)
            k_b = self._batch_unshuffle_ddp(k_b, idx_unshuffle)
            k_grid = self._batch_unshuffle_ddp(k_grid, idx_unshuffle)

        # compute logits
        # FIXME: Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = paddle.sum(q * k, axis=1).unsqueeze(-1)
        # negative logits: NxK
        l_neg = paddle.matmul(q, self.queue.clone().detach())

        # feat point set sim
        backbone_sim_matrix = paddle.matmul(q_b.transpose((0, 2, 1)), k_b)
        densecl_sim_ind = backbone_sim_matrix.argmax(axis=2)  # NxS^2

        gather_index = densecl_sim_ind.unsqueeze(1).expand((-1, k_grid.shape[1], -1))
        indexed_k_grid = paddle_gather(k_grid, dim=2, index=gather_index)
        densecl_sim_q = (q_grid * indexed_k_grid).sum(1)  # NxS^2

        l_pos_dense = densecl_sim_q.reshape((-1, )).unsqueeze(-1) # NS^2X1

        q_grid = q_grid.transpose((0, 2, 1))
        q_grid = q_grid.reshape((-1, q_grid.shape[2]))
        l_neg_dense = paddle.matmul(q_grid, self.queue2.clone().detach())

        loss_single = self.head(l_pos, l_neg)['loss']
        loss_dense = self.head(l_pos_dense, l_neg_dense)['loss']

        outputs = dict()
        outputs['loss'] = loss_single * (1 - self.loss_lambda) + loss_dense * self.loss_lambda

        self._dequeue_and_enqueue(k)
        self._dequeue_and_enqueue2(k2)
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


def paddle_gather(x, dim, index):
    index_shape = index.shape
    index_flatten = index.flatten()
    if dim < 0:
        dim = len(x.shape) + dim
    nd_index = []
    for k in range(len(x.shape)):
        if k == dim:
            nd_index.append(index_flatten)
        else:
            reshape_shape = [1] * len(x.shape)
            reshape_shape[k] = x.shape[k]
            dim_index = paddle.expand(paddle.reshape(paddle.arange(x.shape[k], dtype=index.dtype), reshape_shape),
                                      index_shape).flatten()
            nd_index.append(dim_index)
    ind2 = paddle.transpose(paddle.stack(nd_index), [1, 0])
    paddle_out = paddle.gather_nd(x, ind2).reshape(index_shape)
    return paddle_out


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
