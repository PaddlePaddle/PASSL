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

import copy
import paddle
import paddle.nn as nn

from .builder import BACKBONES

trunc_normal_ = nn.initializer.TruncatedNormal(std=0.02)
zeros_ = nn.initializer.Constant(value=0.0)
ones_ = nn.initializer.Constant(value=1.0)


def drop_path(x, drop_prob=0.0, training=False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ...
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = paddle.to_tensor(1 - drop_prob)
    shape = (paddle.shape(x)[0], ) + (1, ) * (x.ndim - 1)
    random_tensor = keep_prob + paddle.rand(shape, dtype=x.dtype)
    random_tensor = paddle.floor(random_tensor)  # binarize
    output = x.divide(keep_prob) * random_tensor
    return output


class DropPath(nn.Layer):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
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


class PatchEmbedding(nn.Layer):
    def __init__(self,
                 image_size=224,
                 patch_size=4,
                 in_channels=3,
                 embed_dim=96):
        super().__init__()
        image_size = (image_size, image_size)
        patch_size = (patch_size, patch_size)
        patches_resolution = [
            image_size[0] // patch_size[0], image_size[1] // patch_size[1]
        ]
        self.image_size = image_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.patch_embed = nn.Conv2D(in_channels=in_channels,
                                     out_channels=embed_dim,
                                     kernel_size=patch_size,
                                     stride=patch_size)

    def forward(self, x):
        x = self.patch_embed(
            x)  # [batch, embed_dim, h, w] h,w = patch_resolution
        x = x.flatten(start_axis=2,
                      stop_axis=-1)  # [batch, embed_dim, h*w] h*w = num_patches
        x = x.transpose([0, 2, 1])  # [batch, h*w, embed_dim]
        return x


class Mlp(nn.Layer):
    def __init__(self, in_features, hidden_features, dropout=0.):
        super(Mlp, self).__init__()
        w_attr_1, b_attr_1 = self._init_weights()
        self.fc1 = nn.Linear(in_features,
                             hidden_features,
                             weight_attr=w_attr_1,
                             bias_attr=b_attr_1)

        w_attr_2, b_attr_2 = self._init_weights()
        self.fc2 = nn.Linear(hidden_features,
                             in_features,
                             weight_attr=w_attr_2,
                             bias_attr=b_attr_2)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class ClassAttention(nn.Layer):
    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attention_dropout=0.,
                 dropout=0.):
        super().__init__()
        self.num_heads = num_heads
        self.dim_head = dim // num_heads
        self.scale = qk_scale or self.dim_head**-0.5

        self.q = nn.Linear(dim, dim, bias_attr=qkv_bias)
        self.k = nn.Linear(dim, dim, bias_attr=qkv_bias)
        self.v = nn.Linear(dim, dim, bias_attr=qkv_bias)

        self.attn_dropout = nn.Dropout(attention_dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(axis=-1)

    def forward(self, x):
        B, N, C = x.shape

        q = self.q(x[:, :1, :])  # same as x[:, 0], but more intuitive
        q = q.reshape([B, self.num_heads, 1, self.dim_head])

        k = self.k(x)
        k = k.reshape([B, N, self.num_heads, self.dim_head])
        k = k.transpose([0, 2, 1, 3])

        v = self.v(x)
        v = v.reshape([B, N, self.num_heads, self.dim_head])
        v = v.transpose([0, 2, 1, 3])

        attn = paddle.matmul(q * self.scale, k, transpose_y=True)
        attn = self.softmax(attn)
        attn = self.attn_dropout(attn)

        cls_embed = paddle.matmul(attn, v)
        cls_embed = cls_embed.transpose([0, 2, 1, 3])
        cls_embed = cls_embed.reshape([B, 1, C])
        cls_embed = self.proj(cls_embed)
        cls_embed = self.proj_dropout(cls_embed)
        return cls_embed


class TalkingHeadAttention(nn.Layer):
    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 dropout=0.,
                 attention_dropout=0.):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.dim_head = dim // num_heads
        self.scale = self.dim_head**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias_attr=qkv_bias)
        self.attn_dropout = nn.Dropout(attention_dropout)
        self.softmax = nn.Softmax(axis=-1)
        self.proj = nn.Linear(dim, dim)
        self.proj_dropout = nn.Dropout(dropout)

        # talking head
        self.proj_l = nn.Linear(num_heads, num_heads)
        self.proj_w = nn.Linear(num_heads, num_heads)

    def transpose_multihead(self, x):
        new_shape = x.shape[:-1] + [self.num_heads, self.dim_head]
        x = x.reshape(new_shape)
        x = x.transpose([0, 2, 1, 3])
        return x

    def forward(self, x):
        B, H, C = x.shape  # H: num_patches
        qkv = self.qkv(x).chunk(3, axis=-1)
        q, k, v = map(self.transpose_multihead,
                      qkv)  #[B, num_heads, num_patches, single_head_dim]

        q = q * self.scale
        attn = paddle.matmul(
            q, k, transpose_y=True)  #[B, num_heads, num_patches, num_patches]

        # projection across heads (before softmax)
        attn = attn.transpose([0, 2, 3,
                               1])  #[B, num_patches, num_patches, num_heads]
        attn = self.proj_l(attn)
        attn = attn.transpose([0, 3, 1,
                               2])  #[B, num_heads, num_patches, num_patches]

        attn = self.softmax(attn)

        # projection across heads (after softmax)
        attn = attn.transpose([0, 2, 3,
                               1])  #[B, num_patches, num_patches, num_heads]
        attn = self.proj_w(attn)
        attn = attn.transpose([0, 3, 1,
                               2])  #[B, num_heads, num_patches, num_patches]

        attn = self.attn_dropout(attn)

        z = paddle.matmul(attn,
                          v)  #[B, num_heads, num_patches, single_head_dim]
        z = z.transpose([0, 2, 1,
                         3])  #[B, num_patches, num_heads, single_head_dim]

        z = z.reshape([B, H, C])
        z = self.proj(z)
        z = self.proj_dropout(z)

        return z


class LayerScaleBlockClassAttention(nn.Layer):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 dropout=0.,
                 attention_dropout=0.,
                 droppath=0.,
                 init_values=1e-4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, epsilon=1e-6)
        self.attn = ClassAttention(dim,
                                   num_heads=num_heads,
                                   qkv_bias=qkv_bias,
                                   dropout=dropout,
                                   attention_dropout=attention_dropout)
        self.drop_path = DropPath(droppath) if droppath > 0. else Identity()
        self.norm2 = nn.LayerNorm(dim, epsilon=1e-6)
        self.mlp = Mlp(in_features=dim,
                       hidden_features=int(dim * mlp_ratio),
                       dropout=dropout)

        self.gamma_1 = paddle.create_parameter(
            shape=[dim],
            dtype='float32',
            default_initializer=nn.initializer.Constant(init_values))
        self.gamma_2 = paddle.create_parameter(
            shape=[dim],
            dtype='float32',
            default_initializer=nn.initializer.Constant(init_values))

    def forward(self, x, x_cls):
        u = paddle.concat([x_cls, x], axis=1)

        u = self.norm1(u)
        u = self.attn(u)
        u = self.gamma_1 * u
        u = self.drop_path(u)
        x_cls = u + x_cls

        h = x_cls
        x_cls = self.norm2(x_cls)
        x_cls = self.mlp(x_cls)
        x_cls = self.gamma_2 * x_cls
        x_cls = self.drop_path(x_cls)
        x_cls = h + x_cls

        return x_cls


class LayerScaleBlock(nn.Layer):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 dropout=0.,
                 attention_dropout=0.,
                 droppath=0.,
                 init_values=1e-4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, epsilon=1e-6)
        self.attn = TalkingHeadAttention(dim,
                                         num_heads=num_heads,
                                         qkv_bias=qkv_bias,
                                         dropout=dropout,
                                         attention_dropout=attention_dropout)
        self.drop_path = DropPath(droppath) if droppath > 0. else Identity()
        self.norm2 = nn.LayerNorm(dim, epsilon=1e-6)
        self.mlp = Mlp(in_features=dim,
                       hidden_features=int(dim * mlp_ratio),
                       dropout=dropout)

        self.gamma_1 = paddle.create_parameter(
            shape=[dim],
            dtype='float32',
            default_initializer=nn.initializer.Constant(init_values))
        self.gamma_2 = paddle.create_parameter(
            shape=[dim],
            dtype='float32',
            default_initializer=nn.initializer.Constant(init_values))

    def forward(self, x):
        h = x
        x = self.norm1(x)
        x = self.attn(x)
        x = self.gamma_1 * x  #[B, num_patches, embed_dim]
        x = self.drop_path(x)
        x = h + x

        h = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = self.gamma_2 * x  #[B, num_patches, embed_dim]
        x = self.drop_path(x)
        x = h + x
        return x


class Cait(nn.Layer):
    def __init__(self,
                 image_size=224,
                 in_channels=3,
                 num_classes=1000,
                 patch_size=16,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 dropout=0.,
                 attention_dropout=0.,
                 droppath=0,
                 init_values=1e-4,
                 mlp_ratio_class_token=4.0,
                 depth_token_only=2):
        super().__init__()
        self.num_classes = num_classes
        # convert image to paches
        self.patch_embed = PatchEmbedding(image_size=image_size,
                                          patch_size=patch_size,
                                          in_channels=in_channels,
                                          embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        # tokens add for classification
        self.cls_token = paddle.create_parameter(
            shape=[1, 1, embed_dim],
            dtype='float32',
            default_initializer=trunc_normal_)
        # positional embeddings for patch positions
        self.pos_embed = paddle.create_parameter(
            shape=[1, num_patches, embed_dim],
            dtype='float32',
            default_initializer=trunc_normal_)

        self.pos_dropout = nn.Dropout(dropout)

        # create self-attention(layer-scale) layers
        layer_list = []
        for i in range(depth):
            block_layers = LayerScaleBlock(dim=embed_dim,
                                           num_heads=num_heads,
                                           mlp_ratio=mlp_ratio,
                                           qkv_bias=qkv_bias,
                                           dropout=dropout,
                                           attention_dropout=attention_dropout,
                                           droppath=droppath,
                                           init_values=init_values)
            layer_list.append(copy.deepcopy(block_layers))
        self.blocks = nn.LayerList(layer_list)

        # craete class-attention layers
        layer_list = []
        for i in range(depth_token_only):
            block_layers = LayerScaleBlockClassAttention(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio_class_token,
                qkv_bias=qkv_bias,
                dropout=0.,
                attention_dropout=0.,
                droppath=0.,
                init_values=init_values)
            layer_list.append(copy.deepcopy(block_layers))
        self.blocks_token_only = nn.LayerList(layer_list)

        self.norm = nn.LayerNorm(embed_dim, epsilon=1e-6)
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
        # Patch Embedding
        x = self.patch_embed(x)  # [B, num_patches, embed_dim]
        cls_tokens = self.cls_token.expand([x.shape[0], -1,
                                            -1])  # [B, 1, embed_dim]
        x = x + self.pos_embed
        x = self.pos_dropout(x)
        # Self-Attention blocks
        for idx, block in enumerate(self.blocks):
            x = block(x)  # [B, num_patches, embed_dim]
        # Class-Attention blocks
        for idx, block in enumerate(self.blocks_token_only):
            cls_tokens = block(x, cls_tokens)  # [B, 1, embed_dim]
        # Concat outputs
        x = paddle.concat([cls_tokens, x], axis=1)
        x = self.norm(x)  # [B, num_patches + 1, embed_dim]
        return x[:, 0]  # returns only cls_tokens

    def forward(self, x):
        x = self.forward_features(x)
        return x
