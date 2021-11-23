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

# Code was based on https://github.com/microsoft/CvT

from functools import partial

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
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
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
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


class QuickGELU(nn.Layer):
    def forward(self, x: paddle.Tensor):
        return x * F.sigmoid(1.702 * x)


class Attention(nn.Layer):
    def __init__(
        self,
        dim_in,
        dim_out,
        num_heads,
        qkv_bias=False,
        attn_drop=0.0,
        proj_drop=0.0,
        method="dw_bn",
        kernel_size=3,
        stride_kv=1,
        stride_q=1,
        padding_kv=1,
        padding_q=1,
        with_cls_token=True,
        **kwargs,
    ):
        super().__init__()
        self.stride_kv = stride_kv
        self.stride_q = stride_q
        self.dim = dim_out
        self.num_heads = num_heads
        # head_dim = self.qkv_dim // num_heads
        self.scale = dim_out**-0.5
        self.with_cls_token = with_cls_token

        self.conv_proj_q = self._build_projection(
            dim_in,
            dim_out,
            kernel_size,
            padding_q,
            stride_q,
            "linear" if method == "avg" else method,
        )
        self.conv_proj_k = self._build_projection(dim_in, dim_out, kernel_size,
                                                  padding_kv, stride_kv, method)
        self.conv_proj_v = self._build_projection(dim_in, dim_out, kernel_size,
                                                  padding_kv, stride_kv, method)

        self.proj_q = nn.Linear(dim_in, dim_out, bias_attr=qkv_bias)
        self.proj_k = nn.Linear(dim_in, dim_out, bias_attr=qkv_bias)
        self.proj_v = nn.Linear(dim_in, dim_out, bias_attr=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim_out, dim_out)
        self.proj_drop = nn.Dropout(proj_drop)

    def _build_projection(self, dim_in, dim_out, kernel_size, padding, stride,
                          method):
        if method == "dw_bn":
            proj = nn.Sequential(
                nn.Conv2D(
                    dim_in,
                    dim_in,
                    kernel_size=kernel_size,
                    padding=padding,
                    stride=stride,
                    bias_attr=False,
                    groups=dim_in,
                ),
                nn.BatchNorm2D(dim_in),
            )
        elif method == "avg":
            proj = nn.AvgPool2D(kernel_size=kernel_size,
                                padding=padding,
                                stride=stride,
                                ceil_mode=True)

        elif method == "linear":
            proj = None
        else:
            raise ValueError("Unknown method ({})".format(method))

        return proj

    def forward_conv(self, x, h, w):
        if self.with_cls_token:
            cls_token, x = paddle.split(x, [1, h * w], 1)

        x = x.reshape([0, h, w, -1]).transpose([0, 3, 1, 2])

        if self.conv_proj_q is not None:
            q = self.conv_proj_q(x)
            q = q.reshape([0, 0, -1]).transpose([0, 2, 1])

        else:

            q = x.reshape([0, 0, -1]).transpose([0, 2, 1])

        if self.conv_proj_k is not None:
            k = self.conv_proj_k(x)
            k = k.reshape([0, 0, -1]).transpose([0, 2, 1])

        else:

            k = x.reshape([0, 0, -1]).transpose([0, 2, 1])

        if self.conv_proj_v is not None:
            v = self.conv_proj_v(x)
            v = v.reshape([0, 0, -1]).transpose([0, 2, 1])

        else:

            v = x.reshape([0, 0, -1]).transpose([0, 2, 1])

        if self.with_cls_token:
            q = paddle.concat([cls_token, q], axis=1)
            k = paddle.concat([cls_token, k], axis=1)
            v = paddle.concat([cls_token, v], axis=1)

        return q, k, v

    def forward(self, x, h, w):
        if (self.conv_proj_q is not None or self.conv_proj_k is not None
                or self.conv_proj_v is not None):
            q, k, v = self.forward_conv(x, h, w)

        q = self.proj_q(q).reshape([0, 0, self.num_heads,
                                    -1]).transpose([0, 2, 1, 3])
        k = self.proj_k(k).reshape([0, 0, self.num_heads,
                                    -1]).transpose([0, 2, 1, 3])
        v = self.proj_v(v).reshape([0, 0, self.num_heads,
                                    -1]).transpose([0, 2, 1, 3])

        attn_score = paddle.matmul(q, k, transpose_y=True) * self.scale

        attn = F.softmax(attn_score, axis=-1)
        attn = self.attn_drop(attn)

        x = paddle.matmul(attn, v)

        x = x.transpose([0, 2, 1, 3]).reshape([0, 0, -1])

        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Layer):
    def __init__(
        self,
        dim_in,
        dim_out,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        **kwargs,
    ):
        super().__init__()

        self.with_cls_token = kwargs["with_cls_token"]

        self.norm1 = norm_layer(dim_in)
        self.attn = Attention(dim_in, dim_out, num_heads, qkv_bias, attn_drop,
                              drop, **kwargs)

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else Identity()
        self.norm2 = norm_layer(dim_out)

        dim_mlp_hidden = int(dim_out * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim_out,
            hidden_features=dim_mlp_hidden,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x, h, w):
        res = x

        x = self.norm1(x)

        attn = self.attn(x, h, w)
        x = res + self.drop_path(attn)

        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class ConvEmbed(nn.Layer):
    """ Image to Conv Embedding
    """
    def __init__(
        self,
        patch_size=7,
        in_chans=3,
        embed_dim=64,
        stride=4,
        padding=2,
        norm_layer=None,
    ):
        super().__init__()
        patch_size = (patch_size, patch_size)
        self.patch_size = patch_size

        self.proj = nn.Conv2D(in_chans,
                              embed_dim,
                              kernel_size=patch_size,
                              stride=stride,
                              padding=padding)
        self.norm = norm_layer(embed_dim) if norm_layer else None

    def forward(self, x):
        x = self.proj(x)

        B, C, H, W = x.shape

        x = x.transpose([0, 2, 3, 1]).reshape([B, H * W, C])
        if self.norm:
            x = self.norm(x)

        x = x.transpose([0, 2, 1]).reshape([B, C, H, W])

        return x


class VisionTransformer(nn.Layer):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(
        self,
        patch_size=16,
        patch_stride=16,
        patch_padding=0,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        init="trunc_norm",
        **kwargs,
    ):
        super().__init__()
        self.num_features = (
            self.embed_dim
        ) = embed_dim  # num_features for consistency with other models

        self.rearrage = None

        self.patch_embed = ConvEmbed(
            patch_size=patch_size,
            in_chans=in_chans,
            stride=patch_stride,
            padding=patch_padding,
            embed_dim=embed_dim,
            norm_layer=norm_layer,
        )

        with_cls_token = kwargs["with_cls_token"]
        if with_cls_token:

            self.cls_token = paddle.create_parameter(
                shape=[1, 1, embed_dim],
                dtype="float32",
                default_initializer=trunc_normal_,
            )

        else:
            self.cls_token = None

        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in paddle.linspace(0, drop_path_rate, depth)
               ]  # stochastic depth decay rule

        blocks = []
        for j in range(depth):
            blocks.append(
                Block(
                    dim_in=embed_dim,
                    dim_out=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[j],
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    **kwargs,
                ))
        self.blocks = nn.LayerList(blocks)

        self.apply(self._init_weights_trunc_normal)

    def _init_weights_trunc_normal(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight)
            if m.bias is not None:
                zeros_(m.bias)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2D)):
            zeros_(m.bias)
            ones_(m.weight)

    def forward(self, x):
        x = self.patch_embed(x)
        B, C, H, W = x.shape

        x = x.reshape([0, 0, -1]).transpose([0, 2, 1])

        cls_tokens = None
        if self.cls_token is not None:

            cls_tokens = self.cls_token.expand([B, -1, -1])
            x = paddle.concat([cls_tokens, x], axis=1)

        x = self.pos_drop(x)

        for i, blk in enumerate(self.blocks):
            x = blk(x, H, W)

        if self.cls_token is not None:
            cls_tokens, x = paddle.split(x, [1, H * W], 1)

        x = x.reshape([0, H, W, -1]).transpose([0, 3, 1, 2])

        return x, cls_tokens


@BACKBONES.register()
class CvT(nn.Layer):
    def __init__(
        self,
        in_chans=3,
        num_classes=1000,
        act_layer=QuickGELU,
        norm_layer=nn.LayerNorm,
        init="trunc_norm",
        num_stages=3,
        patch_size=[7, 3, 3],
        patch_stride=[4, 2, 2],
        patch_padding=[2, 1, 1],
        embed_dim=[64, 192, 384],
        depth=[1, 2, 10],
        num_heads=[1, 3, 6],
        mlp_ratio=[4.0, 4.0, 4.0],
        qkv_bias=[True, True, True],
        drop_rate=[0.0, 0.0, 0.0],
        attn_drop_rate=[0.0, 0.0, 0.0],
        drop_path_rate=[0.0, 0.0, 0.1],
        with_cls_token=[False, False, True],
        method=["dw_bn", "dw_bn", "dw_bn"],
        kernel_size=[3, 3, 3],
        padding_q=[1, 1, 1],
        padding_kv=[1, 1, 1],
        stride_kv=[2, 2, 2],
        stride_q=[1, 1, 1],
    ):
        super().__init__()
        self.num_classes = num_classes

        self.num_stages = num_stages
        for i in range(self.num_stages):
            kwargs = {
                "patch_size": patch_size[i],
                "patch_stride": patch_stride[i],
                "patch_padding": patch_padding[i],
                "embed_dim": embed_dim[i],
                "depth": depth[i],
                "num_heads": num_heads[i],
                "mlp_ratio": mlp_ratio[i],
                "qkv_bias": qkv_bias[i],
                "drop_rate": drop_rate[i],
                "attn_drop_rate": attn_drop_rate[i],
                "drop_path_rate": drop_path_rate[i],
                "with_cls_token": with_cls_token[i],
                "method": method[i],
                "kernel_size": kernel_size[i],
                "padding_q": padding_q[i],
                "padding_kv": padding_kv[i],
                "stride_kv": stride_kv[i],
                "stride_q": stride_q[i],
            }

            stage = VisionTransformer(
                in_chans=in_chans,
                init=init,
                act_layer=act_layer,
                norm_layer=norm_layer,
                **kwargs,
            )
            setattr(self, f"stage{i}", stage)

            in_chans = embed_dim[i]

        dim_embed = embed_dim[-1]
        self.norm = norm_layer(dim_embed)
        self.cls_token = with_cls_token[-1]

    def forward_features(self, x):

        for i in range(self.num_stages):
            x, cls_tokens = getattr(self, f"stage{i}")(x)

        if self.cls_token:
            x = self.norm(cls_tokens)
            x = paddle.squeeze(x)
        else:
            x = x.reshape([0, 0, -1]).transpose([0, 2, 1])
            x = self.norm(x)
            x = paddle.mean(x, axis=1)

        return x

    def forward(self, x):
        x = self.forward_features(x)

        return x
