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


import math
import paddle
import paddle.nn as nn
from functools import partial, reduce
from operator import mul

from passl.models.base_model import Model
from passl.models.vision_transformer import VisionTransformer, PatchEmbed, to_2tuple
from passl.nn import init
from passl.models.utils.averaged_model import CosineEMA


class MoCoV3ViT(VisionTransformer):
    def __init__(self, stop_grad_conv1=False, **kwargs):
        super().__init__(**kwargs)
        # Use fixed 2D sin-cos position embedding
        self.build_2d_sincos_position_embedding()

        # weight initialization
        for name, m in self.named_sublayers():
            if isinstance(m, nn.Linear):
                if 'qkv' in name:
                    # treat the weights of Q, K, V separately
                    val = math.sqrt(
                        6. / float(m.weight.shape[1] // 3 + m.weight.shape[0]))
                    init.uniform_(m.weight, -val, val)
                else:
                    init.xavier_uniform_(m.weight)
                init.zeros_(m.bias)
        init.normal_(self.cls_token, std=1e-6)

        if isinstance(self.patch_embed, PatchEmbed):
            # xavier_uniform initialization
            val = math.sqrt(6. / float(3 * reduce(
                mul, self.patch_embed.patch_size, 1) + self.embed_dim))
            init.uniform_(self.patch_embed.proj.weight, -val, val)
            init.zeros_(self.patch_embed.proj.bias)

            if stop_grad_conv1:
                self.patch_embed.proj.weight.stop_gradient = True
                self.patch_embed.proj.bias.stop_gradient = True

    def build_2d_sincos_position_embedding(self, temperature=10000.):
        h = self.patch_embed.img_size[0] // self.patch_embed.patch_size[0]
        w = self.patch_embed.img_size[1] // self.patch_embed.patch_size[1]
        grid_w = paddle.arange(w, dtype=paddle.float32)
        grid_h = paddle.arange(h, dtype=paddle.float32)
        grid_w, grid_h = paddle.meshgrid(grid_w, grid_h)
        assert self.embed_dim % 4 == 0, 'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
        pos_dim = self.embed_dim // 4
        omega = paddle.arange(pos_dim, dtype=paddle.float32) / pos_dim
        omega = 1. / (temperature**omega)

        out_w = grid_w.flatten()[..., None] @omega[None]
        out_h = grid_h.flatten()[..., None] @omega[None]
        pos_emb = paddle.concat(
            [
                paddle.sin(out_w), paddle.cos(out_w), paddle.sin(out_h),
                paddle.cos(out_h)
            ],
            axis=1)[None, :, :]
        pe_token = paddle.zeros([1, 1, self.embed_dim], dtype=paddle.float32)

        pos_embed = paddle.concat([pe_token, pos_emb], axis=1)
        self.pos_embed = self.create_parameter(shape=pos_embed.shape)
        self.pos_embed.set_value(pos_embed)
        self.pos_embed.stop_gradient = True
        
        
class MoCoV3Pretrain(Model):
    """
    Build a MoCo model with a base encoder, a momentum encoder, and two MLPs
    https://arxiv.org/abs/1911.05722
    """

    def __init__(self, base_encoder, dim=256, mlp_dim=4096, T=1.0, base_momentum=0.01):
        """
        dim: feature dimension (default: 256)
        mlp_dim: hidden dimension in MLPs (default: 4096)
        T: softmax temperature (default: 1.0)
        """
        super().__init__()

        self.T = T

        # build encoders
        self.base_encoder = base_encoder(class_num=mlp_dim)
        self._build_projector_and_predictor_mlps(dim, mlp_dim)
        
        # create momentum model
        self.momentum_encoder = CosineEMA(
            nn.Sequential(self.base_encoder, self.predictor), momentum=base_momentum)

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
        hidden_dim = self.base_encoder.head.weight.shape[0]
        del self.base_encoder.head  # remove original fc layer

        # projectors
        self.base_encoder.head = self._build_mlp(3, hidden_dim, mlp_dim, dim)

        # predictor
        self.predictor = self._build_mlp(2, dim, mlp_dim, dim)
        
        
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

    def contrastive_loss(self, q, k):
        # normalize
        q = nn.functional.normalize(q, axis=1)
        k = nn.functional.normalize(k, axis=1)
        # gather all targets
        k = self.concat_all_gather(k)
        # Einstein sum is more intuitive
        logits = paddle.einsum('nc,mc->nm', q, k) / self.T
        N = logits.shape[0]  # batch size per GPU
        labels = (paddle.arange(
            N, dtype=paddle.int64) + N * paddle.distributed.get_rank())
        return nn.CrossEntropyLoss()(logits, labels) * (2 * self.T)

    def forward(self, inputs):
        """
        Input:
            x1: first views of images
            x2: second views of images
            m: moco momentum
        Output:
            loss
        """
        
        assert isinstance(inputs, list)
        x1 = inputs[0]
        x2 = inputs[1]

        # compute features
        q1 = self.predictor(self.base_encoder(x1))
        q2 = self.predictor(self.base_encoder(x2))

        with paddle.no_grad():  # no gradient
            # update momentum encoder
            self.momentum_encoder.update_parameters(
                 nn.Sequential(self.base_encoder, self.predictor))

            # compute momentum features as targets
            k1 = self.momentum_encoder(x1)
            k2 = self.momentum_encoder(x2)

        return self.contrastive_loss(q1, k2) + self.contrastive_loss(q2, k1)
        
        
def mocov3_vit_base(**kwargs):
    model = MoCoV3ViT(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(
            nn.LayerNorm, epsilon=1e-6),
        **kwargs)
    return model


def mocov3_pretrain_vit_base(**kwargs):
    base_encoder = partial(mocov3_vit_base, stop_grad_conv1=True)
    model = MoCoV3Pretrain(
        base_encoder=base_encoder,
        dim=256,
        mlp_dim=4096,
        T=0.2,
        base_momentum=0.99,
        **kwargs)
    return model