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

import numpy as np

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import paddle.distributed as dist
import paddle.fluid.layers as layers

from .builder import HEADS


@HEADS.register()
class SwAVHead(nn.Layer):
    """Head for SwAV.

    This head contains clustering and sinkhorn algorithms to compute Q codes.
    """
    def __init__(self,
                 feat_dim,
                 sinkhorn_iterations=3,
                 epsilon=0.05,
                 temperature=0.1,
                 crops_for_assign=[0, 1],
                 num_crops=[2, 6],
                 num_prototypes=3000):
        super(SwAVHead, self).__init__()

        self.sinkhorn_iterations = sinkhorn_iterations
        self.epsilon = epsilon
        self.temperature = temperature
        self.crops_for_assign = crops_for_assign
        self.num_crops = num_crops
        self.use_queue = False
        self.queue = None
        self.world_size = dist.get_world_size()

        # prototype layer
        self.prototypes = None
        if isinstance(num_prototypes, list):
            self.prototypes = MultiPrototypes(feat_dim, num_prototypes)
        elif num_prototypes > 0:
            self.prototypes = nn.Linear(feat_dim, num_prototypes, bias_attr=False)
        assert self.prototypes is not None

    def forward(self, x):
        """Forward head of swav to compute the loss.

        Args:
            x (Tensor): NxC input features.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        # normalize the prototypes
        with paddle.no_grad():
            w = self.prototypes.weight.clone()
            w = F.normalize(w, axis=0, p=2)
            self.prototypes.weight.set_value(w)

        embedding, output = x, self.prototypes(x)
        embedding = embedding.detach()

        bs = int(embedding.shape[0] / sum(self.num_crops))
        loss = 0
        for i, crop_id in enumerate(self.crops_for_assign):
            with paddle.no_grad():
                out = output[bs * crop_id:bs * (crop_id + 1)].detach()
                # time to use the queue
                if self.queue is not None:
                    if self.use_queue or not paddle.all(self.queue[i, -1, :] == 0):
                        self.use_queue = True
                        out = paddle.concat([
                            paddle.mm(self.queue[i],
                                      self.prototypes.weight.t()), out])

                    # fill the queue
                    self.queue[i, bs:] = self.queue[i, :-bs].clone()
                    self.queue[i, :bs] = embedding[crop_id * bs:(crop_id + 1) * bs]

                # get assignments (batch_size * num_prototypes)
                q = distributed_sinkhorn(out, self.sinkhorn_iterations,
                                         self.world_size, self.epsilon)[-bs:]

            # cluster assignment prediction
            subloss = 0
            for v in np.delete(np.arange(np.sum(self.num_crops)), crop_id):
                x = output[bs * v:bs * (v + 1)] / self.temperature
                subloss -= paddle.mean(
                    paddle.sum(q * F.log_softmax(x, axis=1), axis=1))
            loss += subloss / (np.sum(self.num_crops) - 1)
        loss /= len(self.crops_for_assign)
        return dict(loss=loss)


class MultiPrototypes(nn.Layer):
    """Multi-prototypes for SwAV head.

    Args:
        output_dim (int): The output dim from SwAV neck.
        num_prototypes (list[int]): The number of prototypes needed.
    """

    def __init__(self, output_dim, num_prototypes):
        super(MultiPrototypes, self).__init__()
        assert isinstance(num_prototypes, list)
        self.num_heads = len(num_prototypes)
        for i, k in enumerate(num_prototypes):
            self.add_sublayer('prototypes' + str(i),
                               nn.Linear(output_dim, k, bias_attr=False))

    def forward(self, x):
        out = []
        for i in range(self.num_heads):
            out.append(getattr(self, 'prototypes' + str(i))(x))
        return out


@paddle.no_grad()
def distributed_sinkhorn(out, sinkhorn_iterations, world_size, epsilon):
    """Apply the distributed sinknorn optimization on the scores matrix to find
    the assignments."""
    Q = paddle.exp(out / epsilon).t(
    )  # Q is K-by-B for consistency with notations from our paper
    B = Q.shape[1] * world_size  # number of samples to assign
    K = Q.shape[0]  # how many prototypes

    # make the matrix sums to 1
    sum_Q = paddle.sum(Q)
    if dist.get_world_size() > 1:
        dist.all_reduce(sum_Q)
    Q /= sum_Q

    for it in range(sinkhorn_iterations):
        # normalize each row: total weight per prototype must be 1/K
        sum_of_rows = paddle.sum(Q, axis=1, keepdim=True)
        if dist.get_world_size() > 1:
            dist.all_reduce(sum_of_rows)
        Q /= sum_of_rows
        Q /= K

        # normalize each column: total weight per sample must be 1/B
        Q /= paddle.sum(Q, axis=0, keepdim=True)
        Q /= B

    Q *= B  # the columns must sum to 1 so that Q is an assignment
    return Q.t()
