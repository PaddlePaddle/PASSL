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

# Code was based on https://github.com/yitu-opensource/T2T-ViT

import paddle
import paddle.nn as nn
import math
import paddle.nn.functional as F
import numpy as np
from .builder import BACKBONES

trunc_normal_ = nn.initializer.TruncatedNormal(std=0.02)
zeros_ = nn.initializer.Constant(value=0.0)
ones_ = nn.initializer.Constant(value=1.0)


def drop_path(x, drop_prob=0.0, training=False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = paddle.to_tensor(1 - drop_prob)
    shape = (paddle.shape(x)[0], ) + (1, ) * (x.ndim - 1)
    random_tensor = keep_prob + paddle.rand(shape, dtype=x.dtype)
    random_tensor = paddle.floor(random_tensor)  # binarize
    output = x.divide(keep_prob) * random_tensor
    return output


class DropPath(nn.Layer):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Identity(nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class Mlp(nn.Layer):
    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Layer):
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
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(
            [B, N, 3, self.num_heads,
             C // self.num_heads]).transpose([2, 0, 3, 1, 4])
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose([0, 1, 3, 2])) * self.scale
        attn = F.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose([0, 2, 1, 3]).reshape([B, N, C])
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Layer):
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
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim,
                              num_heads=num_heads,
                              qkv_bias=qkv_bias,
                              qk_scale=qk_scale,
                              attn_drop=attn_drop,
                              proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim,
                       hidden_features=mlp_hidden_dim,
                       act_layer=act_layer,
                       drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class TokenAttn(nn.Layer):
    def __init__(self,
                 dim,
                 num_heads=8,
                 in_dim=None,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.in_dim = in_dim
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, in_dim * 3, bias_attr=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(in_dim, in_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape

        qkv = self.qkv(x).reshape([B, N, 3, self.num_heads,
                                   self.in_dim]).transpose([2, 0, 3, 1, 4])
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q * self.scale) @ k.transpose([0, 1, 3, 2])
        attn = F.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose([0, 2, 1, 3]).reshape([B, N, self.in_dim])
        x = self.proj(x)
        x = self.proj_drop(x)

        # skip connection
        x = v.squeeze(
            1
        ) + x  # because the original x has different size with current x, use v to do skip connection

        return x


class Token_transformer(nn.Layer):
    def __init__(self,
                 dim,
                 in_dim,
                 num_heads,
                 mlp_ratio=1.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = TokenAttn(dim,
                              in_dim=in_dim,
                              num_heads=num_heads,
                              qkv_bias=qkv_bias,
                              qk_scale=qk_scale,
                              attn_drop=attn_drop,
                              proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else Identity()
        self.norm2 = norm_layer(in_dim)
        self.mlp = Mlp(in_features=in_dim,
                       hidden_features=int(in_dim * mlp_ratio),
                       out_features=in_dim,
                       act_layer=act_layer,
                       drop=drop)

    def forward(self, x):
        x = self.attn(self.norm1(x))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class Token_performer(nn.Layer):
    def __init__(self,
                 dim,
                 in_dim,
                 head_cnt=1,
                 kernel_ratio=0.5,
                 dp1=0.1,
                 dp2=0.1):
        super().__init__()
        self.emb = in_dim * head_cnt  # we use 1, so it is no need here
        self.kqv = nn.Linear(dim, 3 * self.emb)
        self.dp = nn.Dropout(dp1)
        self.proj = nn.Linear(self.emb, self.emb)
        self.head_cnt = head_cnt
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(self.emb)
        self.epsilon = 1e-8  # for stable in division

        self.mlp = nn.Sequential(
            nn.Linear(self.emb, 1 * self.emb),
            nn.GELU(),
            nn.Linear(1 * self.emb, self.emb),
            nn.Dropout(dp2),
        )

        self.m = int(self.emb * kernel_ratio)

        self.w = paddle.create_parameter(shape=[self.m, self.emb],
                                         dtype='float32',
                                         default_initializer=trunc_normal_)

    def prm_exp(self, x):

        xd = (x * x).sum(axis=-1, keepdim=True)
        xd = xd.expand([xd.shape[0], xd.shape[1], self.m]) / 2
        wtx = paddle.matmul(x, self.w, transpose_y=True)
        out = paddle.exp(wtx - xd) / math.sqrt(self.m)

        return out

    def single_attn(self, x):
        kqv = self.kqv(x).chunk(3, axis=-1)
        k, q, v = kqv[0], kqv[1], kqv[2]

        qp = self.prm_exp(q)
        kp = self.prm_exp(k)

        D = paddle.matmul(qp, kp.sum(axis=1).unsqueeze(2))

        kptv = paddle.matmul(v, kp, transpose_x=True)

        y = paddle.matmul(qp, kptv, transpose_y=True)
        y = y / (D.expand([D.shape[0], D.shape[1], self.emb]) + 1e-8)

        y = v + self.dp(self.proj(y))

        return y

    def forward(self, x):
        x = self.single_attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class T2T_module(nn.Layer):
    """
    Tokens-to-Token encoding module
    """
    def __init__(self,
                 img_size=224,
                 tokens_type='performer',
                 in_chans=3,
                 embed_dim=768,
                 token_dim=64):
        super().__init__()

        if tokens_type == 'transformer':
            print('adopt transformer encoder for tokens-to-token')
            self.soft_split0 = nn.Unfold(kernel_sizes=7,
                                         strides=4,
                                         paddings=[2, 2])
            self.soft_split1 = nn.Unfold(kernel_sizes=3,
                                         strides=2,
                                         paddings=[1, 1])
            self.soft_split2 = nn.Unfold(kernel_sizes=3,
                                         strides=2,
                                         paddings=[1, 1])

            self.attention1 = Token_transformer(dim=in_chans * 7 * 7,
                                                in_dim=token_dim,
                                                num_heads=1,
                                                mlp_ratio=1.0)
            self.attention2 = Token_transformer(dim=token_dim * 3 * 3,
                                                in_dim=token_dim,
                                                num_heads=1,
                                                mlp_ratio=1.0)
            self.project = nn.Linear(token_dim * 3 * 3, embed_dim)

        elif tokens_type == 'performer':
            print('adopt performer encoder for tokens-to-token')
            self.soft_split0 = nn.Unfold(kernel_sizes=7,
                                         strides=4,
                                         paddings=[2, 2])
            self.soft_split1 = nn.Unfold(kernel_sizes=3,
                                         strides=2,
                                         paddings=[1, 1])
            self.soft_split2 = nn.Unfold(kernel_sizes=3,
                                         strides=2,
                                         paddings=[1, 1])

            self.attention1 = Token_performer(dim=in_chans * 7 * 7,
                                              in_dim=token_dim,
                                              kernel_ratio=0.5)
            self.attention2 = Token_performer(dim=token_dim * 3 * 3,
                                              in_dim=token_dim,
                                              kernel_ratio=0.5)
            self.project = nn.Linear(token_dim * 3 * 3, embed_dim)

        elif tokens_type == 'convolution':

            print('adopt convolution layers for tokens-to-token')
            self.soft_split0 = nn.Conv2D(3,
                                         token_dim,
                                         kernel_size=(7, 7),
                                         stride=(4, 4),
                                         padding=(2, 2))
            self.soft_split1 = nn.Conv2D(token_dim,
                                         token_dim,
                                         kernel_size=(3, 3),
                                         stride=(2, 2),
                                         padding=(1, 1))
            self.project = nn.Conv2d(token_dim,
                                     embed_dim,
                                     kernel_size=(3, 3),
                                     stride=(2, 2),
                                     padding=(1, 1))

        self.num_patches = (img_size // (4 * 2 * 2)) * (img_size // (4 * 2 * 2))

    def forward(self, x):
        # step0: soft split

        x = self.soft_split0(x).transpose([0, 2, 1])

        # iteration1: re-structurization/reconstruction
        x = self.attention1(x)
        B, new_HW, C = x.shape

        x = x.transpose([0, 2, 1]).reshape(
            [B, C, int(np.sqrt(new_HW)),
             int(np.sqrt(new_HW))])
        # iteration1: soft split
        x = self.soft_split1(x).transpose([0, 2, 1])

        # iteration2: re-structurization/reconstruction
        x = self.attention2(x)
        B, new_HW, C = x.shape
        x = x.transpose([0, 2, 1]).reshape(
            [B, C, int(np.sqrt(new_HW)),
             int(np.sqrt(new_HW))])
        # iteration2: soft split
        x = self.soft_split2(x).transpose([0, 2, 1])

        # final tokens
        x = self.project(x)

        return x


@BACKBONES.register()
class T2TViT(nn.Layer):
    def __init__(self,
                 img_size=224,
                 tokens_type='performer',
                 in_chans=3,
                 num_classes=1000,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=3.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_layer=nn.LayerNorm,
                 token_dim=64):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.tokens_to_token = T2T_module(img_size=img_size,
                                          tokens_type=tokens_type,
                                          in_chans=in_chans,
                                          embed_dim=embed_dim,
                                          token_dim=token_dim)
        num_patches = self.tokens_to_token.num_patches

        self.cls_token = paddle.create_parameter(
            shape=[1, 1, embed_dim],
            dtype='float32',
            default_initializer=trunc_normal_)

        self.pos_embed = paddle.create_parameter(
            shape=[1, num_patches + 1, embed_dim],
            dtype='float32',
            default_initializer=nn.initializer.Constant(0.0))

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in paddle.linspace(0, drop_path_rate, depth)
               ]  # stochastic depth decay rule
        self.blocks = nn.LayerList([
            Block(dim=embed_dim,
                  num_heads=num_heads,
                  mlp_ratio=mlp_ratio,
                  qkv_bias=qkv_bias,
                  qk_scale=qk_scale,
                  drop=drop_rate,
                  attn_drop=attn_drop_rate,
                  drop_path=dpr[i],
                  norm_layer=norm_layer) for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            zeros_(m.bias)
            ones_(m.weight)

    def forward_features(self, x):
        B = x.shape[0]
        x = self.tokens_to_token(x)

        cls_tokens = self.cls_token.expand([B, -1, -1])
        x = paddle.concat([cls_tokens, x], axis=1)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x[:, 0]

    def forward(self, x):
        x = self.forward_features(x)
        return x
