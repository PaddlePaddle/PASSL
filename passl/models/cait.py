# copyright (c) 2022 PaddlePaddle Authors. All Rights Reserve.
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
import scipy.special
import paddle
import paddle.nn as nn
from functools import partial

from .vision_transformer import Mlp, PatchEmbed, DropPath

from passl.models.base_model import Model
from passl.nn import init

__all__ = [
    "cait_xxs24_224",
    "cait_xxs24_384",
    "cait_xxs36_224",
    "cait_xxs36_384",
    "cait_xs24_384",
    "cait_s24_224",
    "cait_s24_384",
    "cait_s36_384",
    "cait_m36_384",
    "cait_m48_448",
]


class ClassAttn(nn.Layer):
    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.q = nn.Linear(dim, dim, bias_attr=qkv_bias)
        self.k = nn.Linear(dim, dim, bias_attr=qkv_bias)
        self.v = nn.Linear(dim, dim, bias_attr=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        q = self.q(x[:, 0]).unsqueeze(1).reshape(
            (B, 1, self.num_heads, C // self.num_heads)).transpose(
                (0, 2, 1, 3))
        k = self.k(x).reshape(
            (B, N, self.num_heads, C // self.num_heads)).transpose(
                (0, 2, 1, 3))

        q = q * self.scale
        v = self.v(x).reshape(
            (B, N, self.num_heads, C // self.num_heads)).transpose(
                (0, 2, 1, 3))

        attn = (q @k.transpose((0, 1, 3, 2)))
        attn = nn.functional.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)

        x_cls = (attn @v).transpose((0, 2, 1, 3)).reshape((B, 1, C))
        x_cls = self.proj(x_cls)
        x_cls = self.proj_drop(x_cls)

        return x_cls


class LayerScaleBlockClassAttn(nn.Layer):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 attn_block=ClassAttn,
                 mlp_block=Mlp,
                 init_values=1e-4):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = attn_block(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop)
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = mlp_block(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop)
        self.gamma_1 = self.create_parameter(shape=(dim, ))
        self.gamma_2 = self.create_parameter(shape=(dim, ))
        init.constant_(self.gamma_1, init_values)
        init.constant_(self.gamma_2, init_values)

    def forward(self, x, x_cls):
        u = paddle.concat((x_cls, x), axis=1)
        x_cls = x_cls + self.drop_path(self.gamma_1 * self.attn(self.norm1(u)))
        x_cls = x_cls + self.drop_path(self.gamma_2 *
                                       self.mlp(self.norm2(x_cls)))
        return x_cls


class TalkingHeadAttn(nn.Layer):
    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.):
        super().__init__()

        self.num_heads = num_heads

        head_dim = dim // num_heads

        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias_attr=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Linear(dim, dim)

        self.proj_l = nn.Linear(num_heads, num_heads)
        self.proj_w = nn.Linear(num_heads, num_heads)

        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(
            (B, N, 3, self.num_heads, C // self.num_heads)).transpose(
                (2, 0, 3, 1, 4))
        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]

        attn = (q @k.transpose((0, 1, 3, 2)))

        attn = self.proj_l(attn.transpose((0, 2, 3, 1))).transpose(
            (0, 3, 1, 2))

        attn = nn.functional.softmax(attn, axis=-1)

        attn = self.proj_w(attn.transpose((0, 2, 3, 1))).transpose(
            (0, 3, 1, 2))
        attn = self.attn_drop(attn)

        x = (attn @v).transpose((0, 2, 1, 3)).reshape((B, N, C))
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class LayerScaleBlock(nn.Layer):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 attn_block=TalkingHeadAttn,
                 mlp_block=Mlp,
                 init_values=1e-4):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = attn_block(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop)
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = mlp_block(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop)
        self.gamma_1 = self.create_parameter(shape=(dim, ))
        self.gamma_2 = self.create_parameter(shape=(dim, ))
        init.constant_(self.gamma_1, init_values)
        init.constant_(self.gamma_2, init_values)

    def forward(self, x):
        x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class Cait(Model):
    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_chans=3,
                 num_classes=1000,
                 global_pool='token',
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 block_layers=LayerScaleBlock,
                 block_layers_token=LayerScaleBlockClassAttn,
                 patch_layer=PatchEmbed,
                 norm_layer=partial(
                     nn.LayerNorm, epsilon=1e-6),
                 act_layer=nn.GELU,
                 attn_block=TalkingHeadAttn,
                 mlp_block=Mlp,
                 init_values=1e-4,
                 attn_block_token_only=ClassAttn,
                 mlp_block_token_only=Mlp,
                 depth_token_only=2,
                 mlp_ratio_token_only=4.0):
        super().__init__()
        assert global_pool in ('', 'token', 'avg')

        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_features = self.embed_dim = embed_dim

        self.patch_embed = patch_layer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim)

        num_patches = self.patch_embed.num_patches

        self.cls_token = self.create_parameter(
            shape=(1, 1, embed_dim), attr=paddle.ParamAttr(name="cls_token"))
        self.pos_embed = self.create_parameter(
            shape=(1, num_patches, embed_dim),
            attr=paddle.ParamAttr(name="pos_embed"))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [drop_path_rate for i in range(depth)]
        self.blocks = nn.LayerList([
            block_layers(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                attn_block=attn_block,
                mlp_block=mlp_block,
                init_values=init_values) for i in range(depth)
        ])

        self.blocks_token_only = nn.LayerList([
            block_layers_token(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio_token_only,
                qkv_bias=qkv_bias,
                drop=0.0,
                attn_drop=0.0,
                drop_path=0.0,
                norm_layer=norm_layer,
                act_layer=act_layer,
                attn_block=attn_block_token_only,
                mlp_block=mlp_block_token_only,
                init_values=init_values) for i in range(depth_token_only)
        ])

        self.norm = norm_layer(embed_dim)

        self.feature_info = [
            dict(
                num_chs=embed_dim, reduction=0, module='head')
        ]
        self.head = nn.Linear(
            embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        init.trunc_normal_(self.pos_embed, std=.02)
        init.trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            init.constant_(m.bias, 0)
            init.constant_(m.weight, 1.0)

    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=None):
        self.num_classes = num_classes
        if global_pool is not None:
            assert global_pool in ('', 'token', 'avg')
            self.global_pool = global_pool
        self.head = nn.Linear(
            self.num_features,
            num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for i, blk in enumerate(self.blocks):
            x = blk(x)

        cls_tokens = self.cls_token.expand((x.shape[0], -1, -1))
        for i, blk in enumerate(self.blocks_token_only):
            cls_tokens = blk(x, cls_tokens)

        x = paddle.concat((cls_tokens, x), axis=1)
        x = self.norm(x)
        return x

    def forward_head(self, x, pre_logits: bool=False):
        if self.global_pool:
            x = x[:, 1:].mean(axis=1) if self.global_pool == 'avg' else x[:, 0]
        return x if pre_logits else self.head(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
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
        pos_embed_ckt = param_state_dict['pos_embed']  # [1, N, 1024]
        embedding_size = pos_embed_ckt.shape[-1]
        num_patches = self.patch_embed.num_patches
        num_extra_tokens = self.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_ckt.shape[-2] - num_extra_tokens)**0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches**0.5)
        # class_token and dist_token are kept unchanged
        extra_tokens = pos_embed_ckt[:, :num_extra_tokens]
        # only the position tokens are interpolated
        pos_tokens = pos_embed_ckt[0, num_extra_tokens:]
        pos_tokens = paddle.transpose(
            pos_tokens.reshape([-1, orig_size, orig_size, embedding_size]),
            perm=[0, 3, 1, 2])
        dtype = pos_tokens.dtype
        pos_tokens = paddle.nn.functional.interpolate(
            pos_tokens.astype(paddle.float32),
            size=(new_size, new_size),
            mode='bicubic',
            align_corners=False).astype(dtype)
        pos_tokens = paddle.transpose(
            pos_tokens, perm=[0, 2, 3, 1]).flatten(1, 2)
        new_pos_embed = paddle.concat((extra_tokens, pos_tokens), axis=1)
        param_state_dict['pos_embed'] = new_pos_embed

        self.set_dict(param_state_dict)
        return

    def save(self, path, local_rank=0, rank=0):
        paddle.save(self.state_dict(), path + ".pdparams")


def cait_xxs24_224(**kwargs):
    model = Cait(
        img_size=224,
        patch_size=16,
        embed_dim=192,
        depth=24,
        num_heads=4,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(
            nn.LayerNorm, epsilon=1e-6),
        init_values=1e-5,
        depth_token_only=2,
        **kwargs)
    return model


def cait_xxs24_384(**kwargs):
    model = Cait(
        img_size=384,
        patch_size=16,
        embed_dim=192,
        depth=24,
        num_heads=4,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(
            nn.LayerNorm, epsilon=1e-6),
        init_values=1e-5,
        depth_token_only=2,
        **kwargs)

    return model


def cait_xxs36_224(**kwargs):
    model = Cait(
        img_size=224,
        patch_size=16,
        embed_dim=192,
        depth=36,
        num_heads=4,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(
            nn.LayerNorm, epsilon=1e-6),
        init_values=1e-5,
        depth_token_only=2,
        **kwargs)

    return model


def cait_xxs36_384(**kwargs):
    model = Cait(
        img_size=384,
        patch_size=16,
        embed_dim=192,
        depth=36,
        num_heads=4,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(
            nn.LayerNorm, epsilon=1e-6),
        init_values=1e-5,
        depth_token_only=2,
        **kwargs)

    return model


def cait_xs24_384(**kwargs):
    model = Cait(
        img_size=384,
        patch_size=16,
        embed_dim=288,
        depth=24,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(
            nn.LayerNorm, epsilon=1e-6),
        init_values=1e-5,
        depth_token_only=2,
        **kwargs)

    return model


def cait_s24_224(**kwargs):
    model = Cait(
        img_size=224,
        patch_size=16,
        embed_dim=384,
        depth=24,
        num_heads=8,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(
            nn.LayerNorm, epsilon=1e-6),
        init_values=1e-5,
        depth_token_only=2,
        **kwargs)

    return model


def cait_s24_384(**kwargs):
    model = Cait(
        img_size=384,
        patch_size=16,
        embed_dim=384,
        depth=24,
        num_heads=8,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(
            nn.LayerNorm, epsilon=1e-6),
        init_values=1e-5,
        depth_token_only=2,
        **kwargs)

    return model


def cait_s36_384(**kwargs):
    model = Cait(
        img_size=384,
        patch_size=16,
        embed_dim=384,
        depth=36,
        num_heads=8,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(
            nn.LayerNorm, epsilon=1e-6),
        init_values=1e-6,
        depth_token_only=2,
        **kwargs)

    return model


def cait_m36_384(**kwargs):
    model = Cait(
        img_size=384,
        patch_size=16,
        embed_dim=768,
        depth=36,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(
            nn.LayerNorm, epsilon=1e-6),
        init_values=1e-6,
        depth_token_only=2,
        **kwargs)

    return model


def cait_m48_448(**kwargs):
    model = Cait(
        img_size=448,
        patch_size=16,
        embed_dim=768,
        depth=48,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(
            nn.LayerNorm, epsilon=1e-6),
        init_values=1e-6,
        depth_token_only=2,
        **kwargs)

    return model
