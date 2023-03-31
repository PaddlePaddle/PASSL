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

from passl.models.vision_transformer import VisionTransformer, PatchEmbed, to_2tuple
from passl.nn import init


class VisionTransformerMoCo(VisionTransformer):
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


class ConvStem(nn.Layer):
    """
    ConvStem, from Early Convolutions Help Transformers See Better, Tete et al. https://arxiv.org/abs/2106.14881
    """

    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_chans=3,
                 embed_dim=768,
                 norm_layer=None,
                 flatten=True):
        super().__init__()

        assert patch_size == 16, 'ConvStem only supports patch size of 16'
        assert embed_dim % 8 == 0, 'Embed dimension must be divisible by 8 for ConvStem'

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0],
                          img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        # build stem, similar to the design in https://arxiv.org/abs/2106.14881
        stem = []
        input_dim, output_dim = 3, embed_dim // 8
        for l in range(4):
            stem.append(
                nn.Conv2D(
                    input_dim,
                    output_dim,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias_attr=False))
            stem.append(nn.BatchNorm2D(output_dim))
            stem.append(nn.ReLU())
            input_dim = output_dim
            output_dim *= 2
        stem.append(nn.Conv2D(input_dim, embed_dim, kernel_size=1))
        self.proj = nn.Sequential(*stem)

        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose((0, 2, 1))  # BCHW -> BNC
        x = self.norm(x)
        return x


def moco_vit_small(**kwargs):
    model = VisionTransformerMoCo(
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(
            nn.LayerNorm, epsilon=1e-6),
        **kwargs)
    return model


def moco_vit_base(**kwargs):
    model = VisionTransformerMoCo(
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


def moco_vit_conv_small(**kwargs):
    # minus one ViT block
    model = VisionTransformerMoCo(
        patch_size=16,
        embed_dim=384,
        depth=11,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(
            nn.LayerNorm, epsilon=1e-6),
        embed_layer=ConvStem,
        **kwargs)
    return model


def moco_vit_conv_base(**kwargs):
    # minus one ViT block
    model = VisionTransformerMoCo(
        patch_size=16,
        embed_dim=768,
        depth=11,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(
            nn.LayerNorm, epsilon=1e-6),
        embed_layer=ConvStem,
        **kwargs)
    return model
