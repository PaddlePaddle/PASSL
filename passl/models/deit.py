# copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Code was heavily based on https://github.com/facebookresearch/deit

import warnings
import os
import math
from collections.abc import Callable
import numpy as np
import paddle
import paddle.nn as nn

from passl.models.base_model import Model
from passl.nn import init

from .vision_transformer import to_2tuple, Mlp, PatchEmbed, DropPath, Attention, Block

__all__ = [
    "DeiT_tiny_patch16_224",
    "DeiT_small_patch16_224",
    "DeiT_base_patch16_224",
    "DeiT_tiny_distilled_patch16_224",
    "DeiT_small_distilled_patch16_224",
    "DeiT_base_distilled_patch16_224",
    "DeiT_base_patch16_384",
    "DeiT_base_distilled_patch16_384",
    "DeitVisionTransformer",
    "DistilledVisionTransformer",
]


class DeitVisionTransformer(Model):
    """ Vision Transformer with support for patch input
    """

    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_chans=3,
                 class_num=1000,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_layer='nn.LayerNorm',
                 epsilon=1e-5,
                 **kwargs):
        super().__init__()
        self.class_num = class_num

        self.num_features = self.embed_dim = embed_dim

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.pos_embed = self.create_parameter(
            shape=(1, num_patches + 1, embed_dim),
            attr=paddle.ParamAttr(name="pos_embed"))
        self.cls_token = self.create_parameter(
            shape=(1, 1, embed_dim), attr=paddle.ParamAttr(name="cls_token"))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = np.linspace(0, drop_path_rate, depth)

        self.blocks = nn.LayerList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                epsilon=epsilon) for i in range(depth)
        ])

        self.norm = eval(norm_layer)(embed_dim, epsilon=epsilon)

        # Classifier head
        self.head = nn.Linear(embed_dim,
                              class_num) if class_num > 0 else nn.Identity()

        init.trunc_normal_(self.pos_embed, std=.002)
        init.trunc_normal_(self.cls_token, std=.002)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            init.trunc_normal_(m.weight, std=.002)
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            init.constant_(m.bias, 0)
            init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        # B = x.shape[0]
        B = paddle.shape(x)[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand((B, -1, -1))
        x = paddle.concat((cls_tokens, x), axis=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x[:, 0]

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

    def load_pretrained(self, path, rank=0, finetune=False):
        if not os.path.exists(path + '.pdparams'):
            raise ValueError("Model pretrain path {} does not "
                             "exists.".format(path))

        state_dict = self.state_dict()
        param_state_dict = paddle.load(path + ".pdparams")

        # for FP16 saving pretrained weight
        for key, value in param_state_dict.items():
            if key in param_state_dict and key in state_dict and param_state_dict[
                    key].dtype != state_dict[key].dtype:
                param_state_dict[key] = param_state_dict[key].astype(
                    state_dict[key].dtype)

        if not finetune:
            self.set_dict(param_state_dict)
            return

        for k in ['head.weight', 'head.bias']:
            if k in param_state_dict and param_state_dict[
                    k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del param_state_dict[k]

        # interpolate position embedding
        pos_embed_checkpoint = param_state_dict['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = self.patch_embed.num_patches
        num_extra_tokens = self.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens)**
                        0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches**0.5)
        # class_token and dist_token are kept unchanged
        extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
        # only the position tokens are interpolated
        pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
        pos_tokens = paddle.transpose(
            pos_tokens.reshape([-1, orig_size, orig_size, embedding_size]),
            perm=[0, 3, 1, 2])
        pos_tokens = paddle.nn.functional.interpolate(
            pos_tokens,
            size=(new_size, new_size),
            mode='bicubic',
            align_corners=False)
        pos_tokens = paddle.transpose(
            pos_tokens, perm=[0, 2, 3, 1]).flatten(1, 2)
        new_pos_embed = paddle.concat((extra_tokens, pos_tokens), axis=1)
        param_state_dict['pos_embed'] = new_pos_embed

        self.set_dict(param_state_dict)
        return

    def save(self, path, local_rank=0, rank=0):
        paddle.save(self.state_dict(), path + ".pdparams")


class DistilledVisionTransformer(DeitVisionTransformer):
    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 class_num=1000,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4,
                 qkv_bias=False,
                 norm_layer='nn.LayerNorm',
                 epsilon=1e-5,
                 **kwargs):
        super().__init__(
            img_size=img_size,
            patch_size=patch_size,
            class_num=class_num,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            norm_layer=norm_layer,
            epsilon=epsilon,
            **kwargs)
        self.pos_embed = self.create_parameter(
            shape=(1, self.patch_embed.num_patches + 2, self.embed_dim),
            default_initializer=zeros_)
        self.add_parameter("pos_embed", self.pos_embed)

        self.dist_token = self.create_parameter(
            shape=(1, 1, self.embed_dim), default_initializer=zeros_)
        self.add_parameter("cls_token", self.cls_token)

        self.head_dist = nn.Linear(
            self.embed_dim,
            self.class_num) if self.class_num > 0 else nn.Identity()

        init.trunc_normal_(self.dist_token, std=.02)
        init.trunc_normal_(self.pos_embed, std=.02)
        self.head_dist.apply(self._init_weights)

    def forward_features(self, x):
        B = paddle.shape(x)[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand((B, -1, -1))
        dist_token = self.dist_token.expand((B, -1, -1))
        x = paddle.concat((cls_tokens, dist_token, x), axis=1)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x[:, 0], x[:, 1]

    def forward(self, x):
        x, x_dist = self.forward_features(x)
        x = self.head(x)
        x_dist = self.head_dist(x_dist)
        return (x + x_dist) / 2


def DeiT_tiny_patch16_224(**kwargs):
    model = DeitVisionTransformer(
        patch_size=16,
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4,
        qkv_bias=True,
        epsilon=1e-6,
        **kwargs)
    return model


def DeiT_small_patch16_224(**kwargs):
    model = DeitVisionTransformer(
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        epsilon=1e-6,
        **kwargs)
    return model


def DeiT_base_patch16_224(**kwargs):
    model = DeitVisionTransformer(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        epsilon=1e-6,
        **kwargs)
    return model


def DeiT_base_patch16_384(**kwargs):
    model = DeitVisionTransformer(
        img_size=384,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        epsilon=1e-6,
        **kwargs)
    return model


def DeiT_tiny_distilled_patch16_224(**kwargs):
    model = DistilledVisionTransformer(
        patch_size=16,
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4,
        qkv_bias=True,
        epsilon=1e-6,
        **kwargs)
    return model


def DeiT_small_distilled_patch16_224(**kwargs):
    model = DistilledVisionTransformer(
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        epsilon=1e-6,
        **kwargs)
    return model


def DeiT_base_distilled_patch16_224(**kwargs):
    model = DistilledVisionTransformer(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        epsilon=1e-6,
        **kwargs)
    return model


def DeiT_base_distilled_patch16_384(**kwargs):
    model = DistilledVisionTransformer(
        img_size=384,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        epsilon=1e-6,
        **kwargs)
    return model
