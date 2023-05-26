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

import os
import math
from functools import partial

import paddle
import paddle.nn as nn
from paddle.nn.initializer import TruncatedNormal, Normal, Constant

from passl.utils import logger
from passl.models.base_model import Model
from passl.models.vision_transformer import DropPath, Mlp, PatchEmbed

trunc_normal_ = TruncatedNormal(std=.02)
normal_ = Normal(mean=0, std=0.01)
zeros_ = Constant(value=0.)
ones_ = Constant(value=1.)


__all__ = [
    'DINOVisionTransformer',
    'DINO',
    'DINOLinearProbe',
    'dino_deit_small16',
    'dino_deit_small16_linearprobe',
    'dino_deit_small8',
    'dino_deit_small8_linearprobe',
    'dino_vit_base16',
    'dino_vit_base16_linearprobe',
    'dino_vit_base8',
    'dino_vit_base8_linearprobe',
]


class Attention(nn.Layer):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.attn_head_size = int(dim / self.num_heads)
        self.all_head_size = self.attn_head_size * self.num_heads
        self.scales = qk_scale or self.attn_head_size ** -0.5

        # calculate qkv
        self.qkv = nn.Linear(
            dim, self.all_head_size * 3,  # weight for Q K V
            bias_attr=qkv_bias)

        self.proj = nn.Linear(dim, dim)
        self.attn_dropout = nn.Dropout(attn_drop)
        self.proj_dropout = nn.Dropout(proj_drop)

    def transpose_multihead(self, x):
        # input size  [N, ~, embed_dim]
        new_shape = x.shape[0:2] + [self.num_heads, self.attn_head_size]
        # reshape size[N, ~, head, head_size]
        x = x.reshape(new_shape)
        x = x.transpose([0, 2, 1, 3])
        # transpose   [N, head, ~, head_size]
        return x

    def forward(self, x):
        # input x = [N, H * W + 1, embed_dim]
        qkv = self.qkv(x).chunk(3, axis=-1)  # [N, ~, embed_dim * 3]  list
        q, k, v = map(self.transpose_multihead, qkv)  # [N, head, ~, head_size]

        attn = paddle.matmul(q, k, transpose_y=True)  # [N, head, ~, ~]
        attn = attn * self.scales  # softmax(Q*K/(dk^0.5))
        attn = nn.functional.softmax(attn, axis=-1)
        attn = self.attn_dropout(attn)  # [N, head, ~, ~]

        z = paddle.matmul(attn, v)  # [N, head, ~, head_size]
        z = z.transpose([0, 2, 1, 3])  # [N, ~, head, head_size]
        new_shape = z.shape[0:2] + [self.all_head_size]
        z = z.reshape(new_shape)  # [N, ~, embed_dim]
        z = self.proj(z)  # [N, ~, embed_dim]
        z = self.proj_dropout(z)  # [N, ~, embed_dim]
        return z, attn


class Block(nn.Layer):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, return_attention=False):
        y, attn = self.attn(self.norm1(x))
        if return_attention:
            return attn
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class DINOVisionTransformer(nn.Layer):
    """ DINO Vision Transformer """

    def __init__(self, img_size=[224], patch_size=16, in_chans=3, num_classes=0, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, n_last_blocks=1, avgpool_patchtokens=False, **kwargs):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim

        self.patch_embed = PatchEmbed(
            img_size=img_size[0], patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = self.create_parameter(
            shape=[1, 1, embed_dim],
            dtype='float32',
            default_initializer=zeros_)
        trunc_normal_(self.cls_token)

        self.pos_embed = self.create_parameter(
            shape=[1, 1 + num_patches, embed_dim],
            dtype='float32',
            default_initializer=zeros_)
        trunc_normal_(self.pos_embed)

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in paddle.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.LayerList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        self.n_last_blocks = n_last_blocks
        self.avgpool_patchtokens = avgpool_patchtokens

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            zeros_(m.bias)
            ones_(m.weight)

    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_embed.patch_size
        h0 = h // self.patch_embed.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape((1, int(math.sqrt(N)), int(math.sqrt(N)), dim)).transpose((0, 3, 1, 2)),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.transpose((0, 2, 3, 1)).reshape((1, -1, dim))
        return paddle.concat((class_pos_embed.unsqueeze(0), patch_pos_embed), axis=1)

    def prepare_tokens(self, x):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)  # patch linear embedding

        # add the [CLS] token to the embed patch tokens
        cls_tokens = self.cls_token.expand((B, -1, -1))
        x = paddle.concat((cls_tokens, x), axis=1)

        # add positional encoding to each token
        x = x + self.interpolate_pos_encoding(x, w, h)

        return self.pos_drop(x)

    def forward(self, x):
        x = self.prepare_tokens(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x[:, 0]

    def get_last_selfattention(self, x):
        x = self.prepare_tokens(x)
        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = blk(x)
            else:
                # return attention of the last block
                return blk(x, return_attention=True)

    def get_intermediate_layers(self, x, n=1):
        x = self.prepare_tokens(x)
        # we return the output tokens from the `n` last blocks
        output = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if len(self.blocks) - i <= n:
                output.append(self.norm(x))
        return output


class DINO(Model):

    def __init__(self, **kwargs):
        super().__init__()
        print(kwargs)
        backbone_config = kwargs['backbone']
        backbone_type = backbone_config.pop("type", None)
        if backbone_type is not None:
            self.backbone = eval(backbone_type)(**backbone_config)
        else:
            AttributeError(f'Backbone type is not assigned, please assign it.')

    def _load_model(self, path, tag):
        path = path + ".pdparams"
        if os.path.isfile(path):
            para_state_dict = paddle.load(path)

            # vit
            model_state_dict = self.state_dict()
            keys = model_state_dict.keys()
            num_params_loaded = 0
            for k in keys:
                if k not in para_state_dict:
                    logger.info("{} is not in pretrained model".format(k))
                elif list(para_state_dict[k].shape) != list(model_state_dict[k]
                                                            .shape):
                    logger.info(
                        "[SKIP] Shape of pretrained params {} doesn't match.(Pretrained: {}, Actual: {})"
                        .format(k, para_state_dict[k].shape, model_state_dict[k]
                                .shape))
                else:
                    # conpact FP16 saving pretrained weight
                    if model_state_dict[k].dtype != para_state_dict[k].dtype:
                        para_state_dict[k] = para_state_dict[k].astype(model_state_dict[k].dtype)
                    model_state_dict[k] = para_state_dict[k]
                    num_params_loaded += 1
            self.set_dict(model_state_dict)
            logger.info("There are {}/{} variables loaded into {} with {}.".format(
                num_params_loaded, len(model_state_dict), tag, path))
        else:
            logger.info("No pretrained weights found in {}".format(path))

    def load_pretrained(self, path, rank=0, finetune=False):
        pass

    def save(self, path, local_rank=0, rank=0):
        paddle.save(self.state_dict(), path + ".pdparams")

    def _freeze_norm(self, layer):
        if isinstance(layer, (nn.layer.norm._BatchNormBase)):
            layer._use_global_stats = True


class LinearClassifier(nn.Layer):
    """Linear layer to train on top of frozen features"""
    def __init__(self, dim, class_num=1000):
        super(LinearClassifier, self).__init__()
        self.linear = nn.Linear(dim, class_num)
        normal_(self.linear.weight)
        zeros_(self.linear.bias)

    def forward(self, x):
        x = x.reshape((x.shape[0], -1))
        return self.linear(x)


class DINOLinearProbe(DINO):

    def __init__(self, class_num=1000, **kwargs):
        super().__init__(**kwargs)
        self.backbone.eval()

        self.n_last_blocks = self.backbone.n_last_blocks
        self.avgpool_patchtokens = self.backbone.avgpool_patchtokens
        embed_dim = self.backbone.embed_dim * (self.n_last_blocks + int(self.avgpool_patchtokens))
        self.linear = LinearClassifier(embed_dim, class_num)

        # freeze all layers but the last fc
        for name, param in self.named_parameters():
            if name not in ['linear.linear.weight', 'linear.linear.bias']:
                param.stop_gradient = True

        # optimize only the linear classifier
        parameters = list(
            filter(lambda p: not p.stop_gradient, self.parameters()))
        assert len(parameters) == 2  # weight, bias

        self.apply(self._freeze_norm)

    def load_pretrained(self, path, rank=0, finetune=False):
        self._load_model(path, 'backbone')

    def forward(self, inp):
        with paddle.no_grad():
            intermediate_output = self.backbone.get_intermediate_layers(inp, self.n_last_blocks)
            output = paddle.concat([x[:, 0] for x in intermediate_output], axis=-1)
            if self.avgpool_patchtokens:
                output = paddle.concat(
                    (output.unsqueeze(-1), paddle.mean(intermediate_output[-1][:, 1:], axis=1).unsqueeze(-1)),
                    axis=-1
                )
                output = output.reshape((output.shape[0], -1))
        output = self.linear(output)

        return output


def dino_deit_small16(patch_size=16, **kwargs):
    model = DINOVisionTransformer(
        patch_size=patch_size, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, epsilon=1e-6), **kwargs)
    return model


def dino_deit_small16_linearprobe(patch_size=16, **kwargs):
    model = DINOLinearProbe(
        patch_size=patch_size,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(
            nn.LayerNorm, epsilon=1e-6),
        n_last_blocks=4,
        avgpool_patchtokens=False,
        **kwargs)
    return model


def dino_deit_small8(patch_size=8, **kwargs):
    model = DINOVisionTransformer(
        patch_size=patch_size, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, epsilon=1e-6), **kwargs)
    return model


def dino_deit_small8_linearprobe(patch_size=8, **kwargs):
    model = DINOLinearProbe(
        patch_size=patch_size,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(
            nn.LayerNorm, epsilon=1e-6),
        n_last_blocks=4,
        avgpool_patchtokens=False,
        **kwargs)
    return model


def dino_vit_base16(patch_size=16, **kwargs):
    model = DINOVisionTransformer(
        patch_size=patch_size, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, epsilon=1e-6), **kwargs)
    return model


def dino_vit_base16_linearprobe(patch_size=16, **kwargs):
    model = DINOLinearProbe(
        patch_size=patch_size,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(
            nn.LayerNorm, epsilon=1e-6),
        n_last_blocks=1,
        avgpool_patchtokens=True,
        **kwargs)
    return model


def dino_vit_base8(patch_size=8, **kwargs):
    model = DINOVisionTransformer(
        patch_size=patch_size, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, epsilon=1e-6), **kwargs)
    return model


def dino_vit_base8_linearprobe(patch_size=8, **kwargs):
    model = DINOLinearProbe(
        patch_size=patch_size,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(
            nn.LayerNorm, epsilon=1e-6),
        n_last_blocks=1,
        avgpool_patchtokens=True,
        **kwargs)
    return model
