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

# Ref: https://github.com/PaddlePaddle/VIMER/blob/main/CAE/models/modeling_cae_modules.py

import math
import time
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from functools import partial

from .vision_transformer import PatchEmbed, DropPath
from passl.models.base_model import Model
from passl.nn import init

__all__ = [
    'CAEPretrain',
    'cae_small_patch16_224_8k_vocab',
    'cae_base_patch16_224_8k_vocab',
    'cae_large_patch16_224_8k_vocab',
]


class CAEMlp(nn.Layer):
    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias_attr=True)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias_attr=True)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        # x = self.drop(x)
        # commit this for the orignal BERT implement
        x = self.fc2(x)
        x = self.drop(x)
        return x


class CAEAttention(nn.Layer):
    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 window_size=None,
                 attn_head_dim=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias_attr=False)
        if qkv_bias:
            self.q_bias = self.create_parameter([all_head_dim])
            self.v_bias = self.create_parameter([all_head_dim])
            init.zeros_(self.q_bias)
            init.zeros_(self.v_bias)
        else:
            self.q_bias = None
            self.v_bias = None

        if window_size:
            self.window_size = window_size
            self.num_relative_distance = (2 * window_size[0] - 1) * (
                2 * window_size[1] - 1) + 3
            self.relative_position_bias_table = self.create_parameter(
                [self.num_relative_distance, num_heads])  # 2*Wh-1 * 2*Ww-1, nH
            init.zeros_(self.relative_position_bias_table)
            # cls to token & token 2 cls & cls to cls

            # get pair-wise relative position index for each token inside the window
            coords_h = paddle.arange(window_size[0])
            coords_w = paddle.arange(window_size[1])
            coords = paddle.stack(paddle.meshgrid([coords_h,
                                                   coords_w]))  # 2, Wh, Ww
            coords_flatten = paddle.flatten(coords, 1)  # 2, Wh*Ww
            relative_coords = coords_flatten[:, :,
                                             None] - coords_flatten[:,
                                                                    None, :]  # 2, Wh*Ww, Wh*Ww
            relative_coords = relative_coords.transpose([1, 2,
                                                         0])  # Wh*Ww, Wh*Ww, 2
            relative_coords[:, :, 0] += window_size[
                0] - 1  # shift to start from 0
            relative_coords[:, :, 1] += window_size[1] - 1
            relative_coords[:, :, 0] *= 2 * window_size[1] - 1
            relative_position_index = \
                paddle.zeros((window_size[0] * window_size[1] + 1, ) * 2, dtype=relative_coords.dtype)
            relative_position_index[1:, 1:] = relative_coords.sum(
                -1)  # Wh*Ww, Wh*Ww
            relative_position_index[0, 0:] = self.num_relative_distance - 3
            relative_position_index[0:, 0] = self.num_relative_distance - 2
            relative_position_index[0, 0] = self.num_relative_distance - 1

            self.register_buffer("relative_position_index",
                                 relative_position_index)
        else:
            self.window_size = None
            self.relative_position_bias_table = None
            self.relative_position_index = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, bool_masked_pos=None, rel_pos_bias=None):

        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            k_bias = paddle.zeros_like(self.v_bias)
            k_bias.stop_gradient = True
            qkv_bias = paddle.concat((self.q_bias, k_bias, self.v_bias))

        qkv = F.linear(x=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape([B, N, 3, self.num_heads, -1]).transpose(
            [2, 0, 3, 1, 4])
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @k.transpose([0, 1, 3, 2]))  # (B, N_head, N, N)

        if self.relative_position_bias_table is not None:
            relative_position_bias = \
                self.relative_position_bias_table[self.relative_position_index.reshape([-1])].reshape([
                    self.window_size[0] * self.window_size[1] + 1,
                    self.window_size[0] * self.window_size[1] + 1, -1])  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.transpose(
                [2, 0, 1])  # nH, Wh*Ww, Wh*Ww
            attn = attn + relative_position_bias.unsqueeze(0)

        if rel_pos_bias is not None:
            attn = attn + rel_pos_bias

        attn = F.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)

        x = (attn @v).transpose([0, 2, 1, 3]).reshape([B, N, -1])
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


'''
Attention with bool_masked_pos argument.
'''


class CAECrossAttention(nn.Layer):
    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 window_size=None,
                 attn_head_dim=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.q = nn.Linear(dim, all_head_dim, bias_attr=False)
        self.k = nn.Linear(dim, all_head_dim, bias_attr=False)
        self.v = nn.Linear(dim, all_head_dim, bias_attr=False)

        if qkv_bias:
            self.q_bias = self.create_parameter([all_head_dim])
            self.v_bias = self.create_parameter([all_head_dim])
            init.zeros_(self.q_bias)
            init.zeros_(self.v_bias)
        else:
            self.q_bias = None
            self.k_bias = None
            self.v_bias = None

        self.window_size = None
        self.relative_position_bias_table = None
        self.relative_position_index = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self,
                x,
                bool_masked_pos=None,
                rel_pos_bias=None,
                k=None,
                v=None):
        B, N, C = x.shape

        if k is None:
            k = x
            v = x
            N_k = N
            N_v = N
        else:
            N_k = k.shape[1]
            N_v = v.shape[1]

        q_bias, k_bias, v_bias = None, None, None
        if self.q_bias is not None:
            q_bias = self.q_bias
            k_bias = paddle.zeros_like(self.v_bias)
            k_bias.stop_gradient = True
            v_bias = self.v_bias

        q = F.linear(x=x, weight=self.q.weight, bias=q_bias)  # (B, N_q, dim)
        k = F.linear(x=k, weight=self.k.weight, bias=k_bias)  # (B, N_k, dim)
        v = F.linear(x=v, weight=self.v.weight, bias=v_bias)

        q = q.reshape([B, N, 1, self.num_heads, -1]).transpose(
            [2, 0, 3, 1, 4]).squeeze(0)  # (B, num_heads, N_q, dim)
        k = k.reshape([B, N_k, 1, self.num_heads, -1]).transpose(
            [2, 0, 3, 1, 4]).squeeze(0)  # (B, num_heads, N_k, dim)
        v = v.reshape([B, N_v, 1, self.num_heads, -1]).transpose(
            [2, 0, 3, 1, 4]).squeeze(0)  # (B, num_heads, N_v, dim)

        q = q * self.scale
        attn = (q @k.transpose([0, 1, 3, 2]))  # (B, N_head, N, N)

        attn = F.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)

        x = (attn @v).transpose([0, 2, 1, 3]).reshape([B, N, -1])
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class CAECrossAttentionSimple(nn.Layer):
    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 window_size=None,
                 attn_head_dim=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim**-0.5

    def forward(self,
                x,
                bool_masked_pos=None,
                rel_pos_bias=None,
                k=None,
                v=None):
        B, N, C = x.shape
        N_k = k.shape[1]
        N_v = v.shape[1]

        q_bias, k_bias, v_bias = None, None, None
        q = x

        q = q.reshape([B, N, 1, self.num_heads, -1]).transpose(
            [2, 0, 3, 1, 4]).squeeze(0)  # (B, num_heads, N_q, dim)
        k = k.reshape([B, N_k, 1, self.num_heads, -1]).transpose(
            [2, 0, 3, 1, 4]).squeeze(0)  # (B, num_heads, N_k, dim)
        v = v.reshape([B, N_v, 1, self.num_heads, -1]).transpose(
            [2, 0, 3, 1, 4]).squeeze(0)  # (B, num_heads, N_v, dim)

        q = q * self.scale
        attn = (q @k.transpose([-2, -1]))  # (B, N_head, N, N)
        attn = attn.softmax(axis=-1)
        x = (attn @v).transpose([1, 2]).reshape([B, N, -1])

        return x


'''
Self-attention block with bool_masked_pos argument
'''


class CAEBlock(nn.Layer):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 init_values=None,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 window_size=None,
                 attn_head_dim=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = CAEAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            window_size=window_size,
            attn_head_dim=attn_head_dim)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = CAEMlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop)

        if init_values > 0:
            self.gamma_1 = self.create_parameter([dim])
            self.gamma_2 = self.create_parameter([dim])
            init.constant_(self.gamma_1, init_values)
            init.constant_(self.gamma_2, init_values)
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x, bool_masked_pos=None, rel_pos_bias=None):
        if self.gamma_1 is None:
            x = x + self.drop_path(
                self.attn(
                    self.norm1(x),
                    bool_masked_pos=bool_masked_pos,
                    rel_pos_bias=rel_pos_bias))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(
                self.norm1(x),
                bool_masked_pos=bool_masked_pos,
                rel_pos_bias=rel_pos_bias))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))

        return x


'''
Cross-attention block with bool_masked_pos argument
'''


class CAEDecoderBlock(nn.Layer):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 init_values=None,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 window_size=None,
                 attn_head_dim=None):
        super().__init__()

        # NOTE: cross attention
        self.norm1_q_cross = norm_layer(dim)
        self.norm1_k_cross = norm_layer(dim)
        self.norm1_v_cross = norm_layer(dim)
        self.norm2_cross = norm_layer(dim)
        self.cross_attn = CAECrossAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            window_size=window_size,
            attn_head_dim=attn_head_dim)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)

        self.mlp_cross = CAEMlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop)

        if init_values > 0:
            self.gamma_1_cross = self.create_parameter([dim])
            self.gamma_2_cross = self.create_parameter([dim])
            init.constant_(self.gamma_1_cross, init_values)
            init.constant_(self.gamma_2_cross, init_values)
        else:
            self.gamma_1_cross = self.create_parameter([dim])
            self.gamma_2_cross = self.create_parameter([dim])
            init.ones_(self.gamma_1_cross)
            init.ones_(self.gamma_2_cross)
            self.gamma_1_cross.stop_gradient = True
            self.gamma_2_cross.stop_gradient = True

    def forward(self, x_q, x_kv, pos_q, pos_k, bool_masked_pos):
        x = x_q + self.drop_path(self.gamma_1_cross * self.cross_attn(
            self.norm1_q_cross(x_q + pos_q),
            bool_masked_pos,
            k=self.norm1_k_cross(x_kv + pos_k),
            v=self.norm1_v_cross(x_kv)))
        x = self.norm2_cross(x)
        x = x + self.drop_path(self.gamma_2_cross * self.mlp_cross(x))

        return x


class CAEDecoderBlockSimple(nn.Layer):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 init_values=None,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 window_size=None,
                 attn_head_dim=None,
                 no_parameter=False):
        super().__init__()

        # NOTE: cross attention
        if no_parameter:
            self.norm1_q_cross = norm_layer(
                dim, weight_attr=False, bias_attr=False)
            self.norm1_k_cross = norm_layer(
                dim, weight_attr=False, bias_attr=False)
            self.norm1_v_cross = norm_layer(
                dim, weight_attr=False, bias_attr=False)
            self.norm2_cross = norm_layer(
                dim, weight_attr=False, bias_attr=False)
            self.cross_attn = CAECrossAttentionSimple(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                proj_drop=drop,
                window_size=window_size,
                attn_head_dim=attn_head_dim)
        else:
            self.norm1_q_cross = norm_layer(dim)
            self.norm1_k_cross = norm_layer(dim)
            self.norm1_v_cross = norm_layer(dim)
            self.norm2_cross = norm_layer(dim)
            self.cross_attn = CAECrossAttention(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                proj_drop=drop,
                window_size=window_size,
                attn_head_dim=attn_head_dim)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

    def forward(self,
                x_q,
                x_kv,
                pos_q,
                pos_k,
                bool_masked_pos,
                rel_pos_bias=None):
        x_q = self.norm1_q_cross(x_q + pos_q)
        x_k = self.norm1_k_cross(x_kv + pos_k)
        x_v = self.norm1_v_cross(x_kv)

        x = self.cross_attn(
            x_q, bool_masked_pos, rel_pos_bias=rel_pos_bias, k=x_k, v=x_v)

        return x


class CAEEncoder(nn.Layer):
    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_chans=3,
                 vocab_size=8192,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_layer=None,
                 init_values=None,
                 attn_head_dim=None,
                 use_abs_pos_emb=True,
                 init_std=0.02,
                 args=None,
                 **kwargs):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        self.num_patches = num_patches

        self.cls_token = self.create_parameter([1, 1, embed_dim])
        if use_abs_pos_emb:
            self.pos_embed = self.create_parameter(
                [1, num_patches + 1, embed_dim])
        elif args.sincos_pos_emb:
            self.pos_embed = self.create_parameter(
                [1, num_patches + 1, embed_dim])
            self.pos_embed.set_value(
                self.build_2d_sincos_position_embedding(
                    embed_dim, use_cls_token=True))
            self.pos_embed.stop_gradient = True  # fixed sin-cos embedding
        else:
            self.pos_embed = None
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in paddle.linspace(0, drop_path_rate, depth)
               ]  # stochastic depth decay rule
        self.blocks = nn.LayerList([
            CAEBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                init_values=init_values,
                window_size=None,
                attn_head_dim=attn_head_dim, ) for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        self.init_std = init_std

        # init for learnable absolute position embedding
        if self.pos_embed is not None and use_abs_pos_emb:
            init.trunc_normal_(self.pos_embed, std=self.init_std)
        init.trunc_normal_(self.cls_token, std=self.init_std)
        self.apply(self._init_weights)
        self.fix_init_weight()

    def build_2d_sincos_position_embedding(self,
                                           embed_dim=768,
                                           temperature=10000.,
                                           use_cls_token=False):
        h, w = self.patch_embed.grid_size
        grid_w = paddle.arange(w, dtype=paddle.float32)
        grid_h = paddle.arange(h, dtype=paddle.float32)
        grid_w, grid_h = paddle.meshgrid(grid_w, grid_h)
        assert embed_dim % 4 == 0, 'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
        pos_dim = embed_dim // 4
        omega = paddle.arange(pos_dim, dtype=paddle.float32) / pos_dim
        omega = 1. / (temperature**omega)
        out_w = paddle.einsum('m,d->md', grid_w.flatten(), omega)
        out_h = paddle.einsum('m,d->md', grid_h.flatten(), omega)
        pos_emb = paddle.concat(
            [
                paddle.sin(out_w), paddle.cos(out_w), paddle.sin(out_h),
                paddle.cos(out_h)
            ],
            axis=1)[None, :, :]

        if use_cls_token:
            pe_token = paddle.zeros([1, 1, embed_dim], dtype=paddle.float32)
            pos_emb = paddle.concat([pe_token, pos_emb], axis=1)
        return pos_emb

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.set_value(param / math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight, layer_id + 1)
            rescale(layer.mlp.fc2.weight, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            init.trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            init.zeros_(m.bias)
            init.ones_(m.weight)
        elif isinstance(m, nn.Conv2D):
            init.trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                init.zeros_(m.bias)

    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_num_layers(self):
        return len(self.blocks)

    def forward_features(self, x, bool_masked_pos):
        x = self.patch_embed(x)
        batch_size, seq_len, dim = x.shape

        cls_tokens = self.cls_token.expand(
            [batch_size, -1,
             -1])  # stole cls_tokens impl from Phil Wang, thanks

        # NOTE: unmasked embeddings
        x_unmasked = x[~bool_masked_pos].reshape(
            [batch_size, -1, dim])  # [bs, _, c]
        x_unmasked = paddle.concat((cls_tokens, x_unmasked), axis=1)

        # NOTE: unmasked position embeddings
        if self.pos_embed is not None:
            pos_embed = self.pos_embed.expand(
                [batch_size, self.num_patches + 1, dim])
            pos_embed_unmasked = pos_embed[:, 1:][~bool_masked_pos].reshape(
                [batch_size, -1, dim])
            pos_embed_unmasked = paddle.concat(
                (pos_embed[:, :1], pos_embed_unmasked), axis=1)
            x_unmasked = x_unmasked + pos_embed_unmasked

        x_unmasked = self.pos_drop(x_unmasked)

        for blk in self.blocks:
            x_unmasked = blk(x_unmasked, bool_masked_pos)

        x_unmasked = self.norm(x_unmasked)

        return x_unmasked

    def forward(self, x, bool_masked_pos, return_all_tokens=False):
        x = self.forward_features(x, bool_masked_pos=bool_masked_pos)
        return x


'''
##########################
transformer regressor and decoder
##########################
'''


class CAERegressorDecoder(nn.Layer):
    def __init__(self,
                 patch_size=16,
                 num_classes=8192,
                 embed_dim=768,
                 depth=6,
                 num_heads=12,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_layer=None,
                 init_values=None,
                 num_patches=196,
                 init_std=0.02,
                 args=None,
                 patch_shape=(14, 14)):
        super().__init__()

        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.patch_size = patch_size
        self.args = args

        dpr = [x.item() for x in paddle.linspace(0, drop_path_rate, depth)
               ]  # stochastic depth decay rule

        self.blocks = nn.LayerList([
            CAEDecoderBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                init_values=init_values) for i in range(depth)
        ])

        dpr = [
            x.item()
            for x in paddle.linspace(0, drop_path_rate,
                                     args.num_decoder_self_attention)
        ]  # stochastic depth decay rule
        self.self_att_blocks = nn.LayerList([
            CAEBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                init_values=init_values)
            for i in range(args.num_decoder_self_attention)
        ])

        self.norm = norm_layer(embed_dim)
        if args.num_decoder_self_attention > 0:
            self.norm2 = norm_layer(embed_dim)
        self.head = nn.Linear(
            embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        self.init_std = init_std

        init.trunc_normal_(self.head.weight, std=self.init_std)
        self.apply(self._init_weights)
        self.fix_init_weight()

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.set_value(param / math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.cross_attn.proj.weight, layer_id + 1)
            rescale(layer.mlp_cross.fc2.weight, layer_id + 1)

        for layer_id, layer in enumerate(self.self_att_blocks):
            rescale(layer.attn.proj.weight, layer_id + 1)
            rescale(layer.mlp.fc2.weight, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            init.trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            init.zeros_(m.bias)
            init.ones_(m.weight)
        elif isinstance(m, nn.Conv2D):
            init.trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                init.zeros_(m.bias)

    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_num_layers(self):
        return len(self.blocks)

    def forward(self, x_masked, x_unmasked, pos_embed_masked,
                pos_embed_unmasked, bool_masked_pos):
        N_unmask_patch = x_unmasked.shape[1]
        N_mask_patch = x_masked.shape[1]
        '''
        latent contextual regressor
        '''
        for blk in self.blocks:
            x_masked = blk(x_masked,
                           paddle.concat(
                               [x_unmasked, x_masked], axis=1),
                           pos_embed_masked,
                           paddle.concat(
                               [pos_embed_unmasked, pos_embed_masked], axis=1),
                           bool_masked_pos)
        x_masked = self.norm(x_masked)
        latent_pred = x_masked
        '''
        decoder block
        '''
        if len(self.self_att_blocks) > 0:
            x_masked = x_masked + pos_embed_masked  # add pos embed
            for blk in self.self_att_blocks:
                x_masked = blk(x_masked)
            x_masked = self.norm2(x_masked)

        logits = self.head(x_masked)

        return logits, latent_pred


class CAEPretrain(Model):
    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_chans=3,
                 vocab_size=8192,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_layer=None,
                 init_values=None,
                 attn_head_dim=None,
                 use_abs_pos_emb=True,
                 init_std=0.02,
                 args=None,
                 **kwargs):
        super().__init__()

        self.encoder = CAEEncoder(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            init_values=init_values,
            attn_head_dim=attn_head_dim,
            use_abs_pos_emb=use_abs_pos_emb,
            init_std=init_std,
            args=args)

        # Forward the masked patch to the teacher network. The teacher network is the same as the student by default.
        self.teacher_network = CAEEncoder(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            init_values=init_values,
            attn_head_dim=attn_head_dim,
            use_abs_pos_emb=use_abs_pos_emb,
            init_std=init_std,
            args=args)

        # detach the teacher model
        for param in self.teacher_network.parameters():
            param.stop_gradient = True  # TODO check param.detach_()

        self.init_std = init_std
        self.args = args
        self.num_patches = self.encoder.patch_embed.num_patches

        self.regressor_and_decoder = CAERegressorDecoder(
            patch_size=patch_size,
            num_classes=args.decoder_num_classes,
            embed_dim=args.decoder_embed_dim,
            depth=args.regressor_depth,
            num_heads=args.decoder_num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            init_values=args.decoder_layer_scale_init_value,
            num_patches=self.num_patches,
            init_std=init_std,
            args=args)

        if args.decoder_embed_dim != embed_dim:
            self.encoder_to_decoder = nn.Linear(
                embed_dim, args.decoder_embed_dim, bias_attr=True)
            self.encoder_to_decoder_norm = norm_layer(args.decoder_embed_dim)
        else:
            self.encoder_to_decoder = None

        self.mask_token = self.create_parameter([1, 1, args.decoder_embed_dim])
        init.trunc_normal_(self.mask_token, std=self.init_std)

        ### init the weight
        self.apply(self._init_weights)

        # copy the params from the student to teacher
        self._init_teacher()

    def _init_teacher(self):
        for t_param, s_param in zip(self.teacher_network.parameters(),
                                    self.encoder.parameters()):
            t_param.set_value(s_param.detach())

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            init.zeros_(m.bias)
            init.ones_(m.weight)

    def _update_ema_variables(self, ema_decay):
        for t_param, s_param in zip(self.teacher_network.parameters(),
                                    self.encoder.parameters()):
            bias = s_param.detach() * (1 - ema_decay)
            t_param.scale_(ema_decay)
            t_param.add_(bias)

    def forward(self, x, bool_masked_pos, return_all_tokens=None):
        # x: [bs, 3, 224, 224]
        # bool_masked_pos: [bs, num_patch * num_patch]
        batch_size = x.shape[0]
        '''
        ##########################
        encoder
        ##########################
        '''
        x_unmasked = self.encoder(
            x, bool_masked_pos=bool_masked_pos)  # [bs, num_visible + 1, C_e]

        if self.encoder_to_decoder is not None:
            x_unmasked = self.encoder_to_decoder(x_unmasked)  # [64, 49, C_d]
            x_unmasked = self.encoder_to_decoder_norm(x_unmasked)
        '''
        Forward the teacher network
        '''
        with paddle.no_grad():
            latent_target = self.teacher_network(
                x, bool_masked_pos=(~bool_masked_pos))
            latent_target = latent_target[:, 1:, :]  # remove class token
            if self.encoder_to_decoder is not None:
                latent_target = self.encoder_to_decoder_norm(
                    self.encoder_to_decoder(latent_target.detach()))

            self._update_ema_variables(self.args.dual_path_ema)
        '''
        ##########################
        latent contextual regressor
        ##########################
        '''
        b, num_visible_plus1, dim = x_unmasked.shape
        num_masked_patches = self.num_patches - (num_visible_plus1 - 1
                                                 )  # number of masked patches

        # generate position embeddings.
        try:
            pos_embed = self.encoder.pos_embed.expand(
                [batch_size, self.num_patches + 1, dim])
        except:
            pos_embed = self.encoder.build_2d_sincos_position_embedding(
                dim, use_cls_token=True).expand(
                    [batch_size, self.num_patches + 1, dim])

        # pos embed for class token.
        pos_cls_token = pos_embed[:, :1]
        ''' masked pos embed, no class token '''
        pos_embed_masked = pos_embed[:, 1:][bool_masked_pos].reshape(
            [batch_size, -1, dim])
        ''' unmasked pos embed, class token is optional '''
        pos_embed_unmasked = pos_embed[:, 1:][~bool_masked_pos].reshape(
            [batch_size, -1, dim])
        ''' remove class token '''
        x_unmasked = x_unmasked[:, 1:, :]
        ''' masked embedding '''
        x_masked = self.mask_token.expand(
            [batch_size, num_masked_patches, -1])  # [b, num_masked, C_d]

        logits, latent_pred = self.regressor_and_decoder(
            x_masked, x_unmasked, pos_embed_masked, pos_embed_unmasked,
            bool_masked_pos)
        logits = logits.reshape(
            [logits.shape[0] * logits.shape[1],
             logits.shape[2]])  # reshape to calculate loss

        return logits, latent_pred, latent_target


class RelativePositionBias(nn.Layer):
    def __init__(self, window_size, num_heads):
        super().__init__()
        self.window_size = window_size
        self.num_relative_distance = (2 * window_size[0] - 1) * (
            2 * window_size[1] - 1) + 3
        self.relative_position_bias_table = self.create_parameter(
            [self.num_relative_distance, num_heads])  # 2*Wh-1 * 2*Ww-1, nH
        init.zeros_(self.relative_position_bias_table)
        # cls to token & token 2 cls & cls to cls

        # get pair-wise relative position index for each token inside the window
        coords_h = paddle.arange(window_size[0])
        coords_w = paddle.arange(window_size[1])
        coords = paddle.stack(paddle.meshgrid([coords_h,
                                               coords_w]))  # 2, Wh, Ww
        coords_flatten = paddle.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :,
                                         None] - coords_flatten[:,
                                                                None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.transpose([1, 2,
                                                     0])  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * window_size[1] - 1
        relative_position_index = \
            paddle.zeros((window_size[0] * window_size[1] + 1,) * 2, dtype=relative_coords.dtype)
        relative_position_index[1:, 1:] = relative_coords.sum(
            -1)  # Wh*Ww, Wh*Ww
        relative_position_index[0, 0:] = self.num_relative_distance - 3
        relative_position_index[0:, 0] = self.num_relative_distance - 2
        relative_position_index[0, 0] = self.num_relative_distance - 1

        self.register_buffer("relative_position_index",
                             relative_position_index)

        # trunc_normal_(self.relative_position_bias_table, std=.02)

    def forward(self):
        relative_position_bias = \
            self.relative_position_bias_table[self.relative_position_index.reshape([-1])].reshape([
                self.window_size[0] * self.window_size[1] + 1,
                self.window_size[0] * self.window_size[1] + 1, -1])  # Wh*Ww,Wh*Ww,nH
        return relative_position_bias.transpose([2, 0, 1])  # nH, Wh*Ww, Wh*Ww


class CAEViTLinearProbe(Model):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_chans=3,
                 num_classes=1000,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_layer=nn.LayerNorm,
                 init_values=None,
                 use_abs_pos_emb=True,
                 use_rel_pos_bias=False,
                 use_shared_rel_pos_bias=False,
                 use_mean_pooling=True,
                 init_scale=0.001,
                 lin_probe=False,
                 args=None):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.use_mean_pooling = use_mean_pooling

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = self.create_parameter([1, 1, embed_dim])
        self.use_abs_pos_emb = use_abs_pos_emb
        if use_abs_pos_emb:
            self.pos_embed = self.create_parameter(
                [1, num_patches + 1, embed_dim])
        elif args.sin_pos_emb:
            # sine-cosine positional embeddings is on the way
            self.pos_embed = self.create_parameter(
                [1, num_patches + 1, embed_dim])
            self.pos_embed.set_value(
                self.build_2d_sincos_position_embedding(embed_dim))
            self.pos_embed.stop_gradient = True  # fixed sin-cos embedding
        else:
            self.pos_embed = None

        self.pos_drop = nn.Dropout(p=drop_rate)

        if use_shared_rel_pos_bias:
            self.rel_pos_bias = RelativePositionBias(
                window_size=self.patch_embed.patch_shape, num_heads=num_heads)
        else:
            self.rel_pos_bias = None

        dpr = [x.item() for x in paddle.linspace(0, drop_path_rate, depth)
               ]  # stochastic depth decay rule
        self.use_rel_pos_bias = use_rel_pos_bias
        self.blocks = nn.LayerList([
            CAEBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                init_values=init_values,
                window_size=self.patch_embed.grid_size
                if use_rel_pos_bias else None) for i in range(depth)
        ])
        self.norm = nn.Identity() if use_mean_pooling else norm_layer(
            embed_dim)

        if self.use_mean_pooling:
            self.fc_norm = norm_layer(
                embed_dim, weight_attr=True, bias_attr=True)

        self.lin_probe = lin_probe
        # NOTE: batch norm
        self.args = args
        self.linear_type = args.linear_type
        if lin_probe:
            if args.linear_type != 'standard':
                if args.linear_type == 'attentive_no_parameter':
                    no_parameter = True
                else:
                    no_parameter = False

                self.linear_blocks = nn.LayerList([
                    CAEDecoderBlockSimple(
                        dim=embed_dim,
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        drop=drop_rate,
                        attn_drop=attn_drop_rate,
                        drop_path=0,
                        norm_layer=norm_layer,
                        init_values=0,
                        no_parameter=no_parameter)
                    for i in range(args.linear_depth)
                ])

                self.query_token = self.create_parameter([1, 1, embed_dim])
                init.trunc_normal_(self.query_token, std=.02)

        self.head = nn.Linear(
            embed_dim, num_classes,
            bias_attr=True) if num_classes > 0 else nn.Identity()

        if self.pos_embed is not None and use_abs_pos_emb:
            init.trunc_normal_(self.pos_embed, std=.02)
        init.trunc_normal_(self.cls_token, std=.02)

        init.trunc_normal_(self.head.weight, std=.02)
        init.zeros_(self.head.bias)
        self.apply(self._init_weights)
        self.fix_init_weight()
        self.head.weight.set_value(self.head.weight * init_scale)
        self.head.bias.set_value(self.head.bias * init_scale)

    def build_2d_sincos_position_embedding(self,
                                           embed_dim=768,
                                           temperature=10000.):
        h, w = self.patch_embed.grid_size
        grid_w = paddle.arange(w, dtype=paddle.float32)
        grid_h = paddle.arange(h, dtype=paddle.float32)
        grid_w, grid_h = paddle.meshgrid(grid_w, grid_h)
        assert embed_dim % 4 == 0, 'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
        pos_dim = embed_dim // 4
        omega = paddle.arange(pos_dim, dtype=paddle.float32) / pos_dim
        omega = 1. / (temperature**omega)
        out_w = paddle.einsum('m,d->md', grid_w.flatten(), omega)
        out_h = paddle.einsum('m,d->md', grid_h.flatten(), omega)
        pos_emb = paddle.concat(
            [
                paddle.sin(out_w), paddle.cos(out_w), paddle.sin(out_h),
                paddle.cos(out_h)
            ],
            axis=1)[None, :, :]

        pe_token = paddle.zeros([1, 1, embed_dim], dtype=paddle.float32)
        _, num_patches, _ = pos_emb.shape
        pos_embed = self.create_parameter(
            shape=[1, 1 + num_patches, embed_dim],
            default_initializer=nn.initializer.Assign(
                paddle.concat(
                    [pe_token, pos_emb], axis=1)))
        pos_embed.stop_gradient = True
        return pos_embed

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.scale(1 / (math.sqrt(2.0 * layer_id)))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight, layer_id + 1)
            rescale(layer.mlp.fc2.weight, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            try:
                init.zeros_(m.bias)
                init.ones_(m.weight)
            except:
                pass

    def get_num_layers(self):
        return len(self.blocks)

    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(
            self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x, is_train=True):
        x = self.patch_embed(x)
        batch_size, seq_len, _ = x.shape

        cls_tokens = self.cls_token.expand(
            [batch_size, -1,
             -1])  # stole cls_tokens impl from Phil Wang, thanks
        x = paddle.concat((cls_tokens, x), axis=1)
        if self.pos_embed is not None:
            if self.use_abs_pos_emb:
                x = x + self.pos_embed.expand(
                    [batch_size, -1, -1]).astype(x.dtype).clone().detach()
            else:
                x = x + self.pos_embed.expand(
                    [batch_size, -1, -1]).astype(x.dtype).clone().detach()

        x = self.pos_drop(x)

        rel_pos_bias = self.rel_pos_bias(
        ) if self.rel_pos_bias is not None else None
        for blk in self.blocks:
            x = blk(x, rel_pos_bias=rel_pos_bias)

        x = self.norm(x)

        if self.linear_type == 'standard':
            if self.use_mean_pooling:
                x = x[:, 1:, :].mean(axis=1)  # global pool without cls token
                outcome = self.fc_norm(x)
                return outcome
            else:
                return x[:, 0]
        else:
            query_tokens = self.query_token.expand([batch_size, -1, -1])
            key_value_pos_embed = self.pos_embed.expand(
                [batch_size, -1, -1]).astype(x.dtype).clone().detach()

            x = x + key_value_pos_embed
            for blk in self.linear_blocks:
                query_tokens = blk(query_tokens,
                                   x,
                                   0,
                                   0,
                                   bool_masked_pos=None,
                                   rel_pos_bias=None)

            return query_tokens[:, 0, :]

    def forward(self, x, is_train=True):
        x = self.forward_features(x, is_train)
        x = self.head(x)
        return x


def cae_small_patch16_224_8k_vocab(**kwargs):
    model = CAEPretrain(
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(
            nn.LayerNorm, epsilon=1e-6, weight_attr=True, bias_attr=True),
        vocab_size=8192,
        **kwargs)
    return model


def cae_base_patch16_224_8k_vocab(**kwargs):
    model = CAEPretrain(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(
            nn.LayerNorm, epsilon=1e-6, weight_attr=True, bias_attr=True),
        vocab_size=8192,
        **kwargs)
    return model


def cae_large_patch16_224_8k_vocab(**kwargs):
    model = CAEPretrain(
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(
            nn.LayerNorm, epsilon=1e-6, weight_attr=True, bias_attr=True),
        vocab_size=8192,
        **kwargs)
    return model


def cae_small_patch16_224(**kwargs):
    model = CAEViTLinearProbe(
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(
            nn.LayerNorm, epsilon=1e-6, weight_attr=True, bias_attr=True),
        **kwargs)
    return model


def cae_base_patch16_224(**kwargs):
    model = CAEViTLinearProbe(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(
            nn.LayerNorm, epsilon=1e-6, weight_attr=True, bias_attr=True),
        **kwargs)
    return model


def cae_base_patch16_384(**kwargs):
    model = CAEViTLinearProbe(
        img_size=384,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(
            nn.LayerNorm, epsilon=1e-6, weight_attr=True, bias_attr=True),
        **kwargs)
    return model


def cae_large_patch16_224(pretrained=False, **kwargs):
    model = CAEViTLinearProbe(
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(
            nn.LayerNorm, epsilon=1e-6, weight_attr=True, bias_attr=True),
        **kwargs)
    return model


def cae_large_patch16_384(pretrained=False, **kwargs):
    model = CAEViTLinearProbe(
        img_size=384,
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(
            nn.LayerNorm, epsilon=1e-6, weight_attr=True, bias_attr=True),
        **kwargs)
    return model


def cae_large_patch16_512(pretrained=False, **kwargs):
    model = CAEViTLinearProbe(
        img_size=512,
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(
            nn.LayerNorm, epsilon=1e-6, weight_attr=True, bias_attr=True),
        **kwargs)
    return model
