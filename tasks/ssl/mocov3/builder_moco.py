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

import paddle
import paddle.nn as nn


class MoCo(nn.Layer):
    """
    Build a MoCo model with a base encoder, a momentum encoder, and two MLPs
    https://arxiv.org/abs/1911.05722
    """

    def __init__(self, base_encoder, dim=256, mlp_dim=4096, T=1.0):
        """
        dim: feature dimension (default: 256)
        mlp_dim: hidden dimension in MLPs (default: 4096)
        T: softmax temperature (default: 1.0)
        """
        super(MoCo, self).__init__()

        self.T = T

        # build encoders
        self.base_encoder = base_encoder(num_classes=mlp_dim)
        self.momentum_encoder = base_encoder(num_classes=mlp_dim)

        self._build_projector_and_predictor_mlps(dim, mlp_dim)

        for param_b, param_m in zip(self.base_encoder.parameters(),
                                    self.momentum_encoder.parameters()):
            param_m.copy_(param_b, False)  # initialize
            param_m.stop_gradient = True  # not update by gradient

    def _build_mlp(self,
                   num_layers,
                   input_dim,
                   mlp_dim,
                   output_dim,
                   last_bn=True):
        mlp = []
        for l in range(num_layers):
            dim1 = input_dim if l == 0 else mlp_dim
            dim2 = output_dim if l == num_layers - 1 else mlp_dim

            mlp.append(nn.Linear(dim1, dim2, bias_attr=False))

            if l < num_layers - 1:
                mlp.append(nn.BatchNorm1D(dim2))
                mlp.append(nn.ReLU())
            elif last_bn:
                # follow SimCLR's design: https://github.com/google-research/simclr/blob/master/model_util.py#L157
                # for simplicity, we further removed gamma in BN
                mlp.append(
                    nn.BatchNorm1D(
                        dim2, weight_attr=False, bias_attr=False))

        return nn.Sequential(*mlp)

    def _build_projector_and_predictor_mlps(self, dim, mlp_dim):
        pass

    @paddle.no_grad()
    def _update_momentum_encoder(self, m):
        """Momentum update of the momentum encoder"""
        with paddle.amp.auto_cast(False):
            for param_b, param_m in zip(self.base_encoder.parameters(),
                                        self.momentum_encoder.parameters()):
                paddle.assign((param_m * m + param_b * (1. - m)), param_m)

    def contrastive_loss(self, q, k):
        # normalize
        q = nn.functional.normalize(q, axis=1)
        k = nn.functional.normalize(k, axis=1)
        # gather all targets
        k = concat_all_gather(k)
        # Einstein sum is more intuitive
        logits = paddle.einsum('nc,mc->nm', q, k) / self.T
        N = logits.shape[0]  # batch size per GPU
        labels = (paddle.arange(
            N, dtype=paddle.int64) + N * paddle.distributed.get_rank())
        return nn.CrossEntropyLoss()(logits, labels) * (2 * self.T)

    def forward(self, x1, x2, m):
        """
        Input:
            x1: first views of images
            x2: second views of images
            m: moco momentum
        Output:
            loss
        """

        # compute features
        q1 = self.predictor(self.base_encoder(x1))
        q2 = self.predictor(self.base_encoder(x2))

        with paddle.no_grad():  # no gradient
            self._update_momentum_encoder(m)  # update the momentum encoder

            # compute momentum features as targets
            k1 = self.momentum_encoder(x1)
            k2 = self.momentum_encoder(x2)

        return self.contrastive_loss(q1, k2) + self.contrastive_loss(q2, k1)


class MoCo_ResNet(MoCo):
    def _build_projector_and_predictor_mlps(self, dim, mlp_dim):
        hidden_dim = self.base_encoder.fc.weight.shape[0]
        del self.base_encoder.fc, self.momentum_encoder.fc  # remove original fc layer

        # projectors
        self.base_encoder.fc = self._build_mlp(2, hidden_dim, mlp_dim, dim)
        self.momentum_encoder.fc = self._build_mlp(2, hidden_dim, mlp_dim, dim)

        # predictor
        self.predictor = self._build_mlp(2, dim, mlp_dim, dim, False)


class MoCo_ViT(MoCo):
    def _build_projector_and_predictor_mlps(self, dim, mlp_dim):
        hidden_dim = self.base_encoder.head.weight.shape[0]
        del self.base_encoder.head, self.momentum_encoder.head  # remove original fc layer

        # projectors
        self.base_encoder.head = self._build_mlp(3, hidden_dim, mlp_dim, dim)
        self.momentum_encoder.head = self._build_mlp(3, hidden_dim, mlp_dim,
                                                     dim)

        # predictor
        self.predictor = self._build_mlp(2, dim, mlp_dim, dim)


# utils
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
