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

import math
import paddle
import paddle.nn as nn

from ...modules.init import init_backbone_weight
from .builder import MODELS
from ..backbones import build_backbone
from ..necks import build_neck
from ..heads import build_head
from ...modules.init import init_backbone_weight, normal_init, kaiming_init, constant_, reset_parameters, xavier_init


class EncoderwithProjection_online(nn.Layer):
    """EncoderwithProjection_online"""
    def __init__(self, backbone, neck):
        super().__init__()
        # backbone
        base_encoder = build_backbone(backbone)
        self.encoder = nn.Sequential(*list(base_encoder.children())[:-1])
        # projection
        self.projection = build_neck(neck)

    def forward(self, x):
        """forward"""
        x = self.encoder(x)
        x = paddle.flatten(x, 1)
        x = self.projection(x)
        return x


class Predictor(nn.Layer):
    """Predictor"""
    def __init__(self, pred):
        super().__init__()

        # predictor
        self.predictor = build_neck(pred)

    def forward(self, x):
        """forward"""
        return self.predictor(x)

@MODELS.register()
class MoCoBYOL(nn.Layer):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self,
                 backbone,
                 neck=None,
                 head=None,
                 predictor=None,
                 queue_dim=65536,
                 T=0.2,
                 dim=256,
                 target_decay_method='fixed',
                 target_decay_rate=0.996,
                 align_init_network=True,
                 use_synch_bn=True):
        """
        Args:
            backbone (dict): config of backbone.
            neck (dict): config of neck.
            head (dict): config of head.
            dim (int): feature dimension. Default: 256.
        """
        super(MoCoBYOL, self).__init__()

        # create the encoders
        # num_classes is the output fc dimension

        # online network
        self.online_network = EncoderwithProjection_online(backbone, neck)

        # target network
        self.target_network = EncoderwithProjection_online(backbone, neck)

        # predictor
        self.predictor = Predictor(predictor)

        self.initializes_target_network()

        #self.towers = nn.LayerList()
        self.base_m = target_decay_rate
        self.target_decay_method = target_decay_method
        
                
        # Convert BatchNorm*d to SyncBatchNorm*d
        if use_synch_bn:
            self.online_network = nn.SyncBatchNorm.convert_sync_batchnorm(self.online_network)
            self.target_network = nn.SyncBatchNorm.convert_sync_batchnorm(self.target_network)
            self.predictor = nn.SyncBatchNorm.convert_sync_batchnorm(self.predictor)

        self.head = build_head(head)

        #moco
        # create the queue
        output_dim = dim 
        k_dim = queue_dim
        self.T = 0.2
        self.K = k_dim 
        self.register_buffer("queue", paddle.randn([output_dim, k_dim]))
        self.queue = nn.functional.normalize(self.queue, axis=0)
        self.register_buffer("queue_ptr", paddle.zeros([1], 'int64'))

    @paddle.no_grad()
    def initializes_target_network(self):
        for param_q, param_k in zip(self.online_network.parameters(), self.target_network.parameters()):
            param_k.set_value(param_q)  # initialize
            param_k.stop_gradient = True

    @paddle.no_grad()
    def update_target_network(self):
        for param_q, param_k in zip(self.online_network.parameters(),
                                    self.target_network.parameters()):
            paddle.assign((param_k * self.m + param_q * (1. - self.m)), param_k)
            param_k.stop_gradient = True
    # moco
    @paddle.no_grad()
    def _dequeue_and_enqueue(self, keys):
        keys = concat_all_gather(keys)
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr[0])
        assert self.K % batch_size == 0  # for simplicity
        self.queue[:, ptr:ptr + batch_size] = keys.transpose([1, 0])
        ptr = (ptr + batch_size) % self.K  # move pointer
        self.queue_ptr[0] = ptr

    def train_iter(self, *inputs, **kwargs):
        current_iter = kwargs['current_iter']
        total_iters =  kwargs['total_iters']

        if self.target_decay_method == 'cosine':
            self.m = 1 - (1 - self.base_m) * (math.cos(math.pi*(current_iter-1)/total_iters) + 1) / 2.0   # 47.0
        elif self.target_decay_method == 'fixed':
            self.m = self.base_m   # 55.7
        else:
            raise NotImplementedError

        # dual imputs
        img_a, img_b = inputs
        
        # online network forward
        q_out = self.online_network(paddle.concat((img_a, img_b), axis=0))
        q = self.predictor(q_out)
        moco_q = q_out[:img_a.shape[0]]
        moco_kk = q_out[img_a.shape[0]:]
        moco_q = nn.functional.normalize(moco_q, axis=1)
        moco_kk = nn.functional.normalize(moco_kk, axis=1)

        # target network forward
        with paddle.no_grad():
            self.update_target_network()
            target_z = self.target_network(paddle.concat((img_b, img_a), axis=0)).detach().clone()
            moco_k = target_z[:img_b.shape[0]]
            moco_qq = target_z[img_b.shape[0]:]
            moco_k = nn.functional.normalize(moco_k, axis=1)
            moco_qq = nn.functional.normalize(moco_qq, axis=1)

        # moco

        l_pos_q = paddle.sum(moco_q * moco_k, axis=1).unsqueeze(-1)
        l_neg_q = paddle.matmul(moco_q, self.queue.clone().detach())
        logits_q = paddle.concat((l_pos_q, l_neg_q), axis=1)
        logits_q /= self.T
        labels_q = paddle.zeros([logits_q.shape[0]], dtype='int64')

        l_pos_k = paddle.sum(moco_kk * moco_qq, axis=1).unsqueeze(-1)
        l_neg_k = paddle.matmul(moco_kk, self.queue.clone().detach())
        logits_k = paddle.concat((l_pos_k, l_neg_k), axis=1)
        logits_k /= self.T
        labels_k = paddle.zeros([logits_k.shape[0]], dtype='int64')

        self._dequeue_and_enqueue(moco_k)
        self._dequeue_and_enqueue(moco_qq)

        return self.head(q, target_z, logits_q, labels_q, logits_k, labels_k)



    def forward(self, *inputs, mode='train', **kwargs):
        if mode == 'train':
            return self.train_iter(*inputs, **kwargs)
        elif mode == 'test':
            return self.test_iter(*inputs, **kwargs)
        elif mode == 'extract':
            return self.backbone(*inputs)
        else:
            raise Exception("No such mode: {}".format(mode))



    # L1 update
    @paddle.no_grad()
    def update_target_network_L1(self):
        for param_q, param_k in zip(self.towers[0].parameters(),
                                    self.towers[1].parameters()):
            paddle.assign(param_k - (1-self.m)*paddle.sign(param_k-param_q), param_k)
            param_k.stop_gradient = True

    # L2 + L1
    @paddle.no_grad()
    def update_target_network_clip(self):
        for param_q, param_k in zip(self.towers[0].parameters(),
                                    self.towers[1].parameters()):
            # paddle.assign((param_k * self.m + param_q * (1. - self.m)), param_k)
            paddle.assign(param_k - (1-self.m) * paddle.clip((param_k - param_q), min=-1.0, max=1.0) , param_k)
            param_k.stop_gradient = True

    @paddle.no_grad()
    def update_target_network_LN_clip(self):
        for param_q, param_k in zip(self.towers[0].parameters(),
                                    self.towers[1].parameters()):
            paddle.assign((param_k * self.m + param_q * (1. - self.m)), param_k)
            paddle.assign(param_k - (1-self.m) * paddle.clip((param_k - param_q), min=-1.0, max=1.0) , param_k)
            param_k.stop_gradient = True



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
