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
import numpy as np
from functools import partial

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn.initializer import TruncatedNormal, Normal, Constant

from passl.nn import init
from passl.utils import logger
from passl.models.base_model import Model
from passl.models.vision_transformer import to_2tuple, DropPath, Mlp, Attention

trunc_normal_ = TruncatedNormal(std=.02)
normal_ = Normal(mean=0, std=0.01)
zeros_ = Constant(value=0.)
ones_ = Constant(value=1.)


__all__ = [
    'DINOv2VisionTransformer',
    'DINOv2',
    'DINOv2LinearProbe',
    'dinov2_vit_small',
    'dinov2_vit_small_linearprobe',
    'dinov2_vit_base',
    'dinov2_vit_base_linearprobe',
    'dinov2_vit_large',
    'dinov2_vit_large_linearprobe',
    'dinov2_vit_gaint2',
    'dinov2_vit_gaint2_linearprobe',
]


class LayerScale(nn.Layer):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = self.create_parameter(
            shape=[dim], default_initializer=paddle.nn.initializer.Constant(value=init_values))

    def forward(self, x):
        return x.scale_(self.gamma) if self.inplace else x * self.gamma


class Block(nn.Layer):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        ffn_bias=True,
        drop=0.0,
        attn_drop=0.0,
        init_values=None,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        attn_class=Attention,
        ffn_layer=Mlp):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = attn_class(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ffn_layer(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.sample_drop_ratio = drop_path

    def forward(self, x):
        def attn_residual_func(x):
            return self.ls1(self.attn(self.norm1(x)))

        def ffn_residual_func(x):
            return self.ls2(self.mlp(self.norm2(x)))

        if self.training and self.sample_drop_ratio > 0.1:
            # the overhead is compensated only for a drop path rate larger than 0.1
            x = drop_add_residual_stochastic_depth(
                x,
                residual_func=attn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
            )
            x = drop_add_residual_stochastic_depth(
                x,
                residual_func=ffn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
            )
        elif self.training and self.sample_drop_ratio > 0.0:
            x = x + self.drop_path1(attn_residual_func(x))
            x = x + self.drop_path1(ffn_residual_func(x))  # FIXME: drop_path2
        else:
            x = x + attn_residual_func(x)
            x = x + ffn_residual_func(x)
        return x


def drop_add_residual_stochastic_depth(x, residual_func, sample_drop_ratio=0.0):
    # 1) extract subset using permutation
    b, n, d = x.shape
    sample_subset_size = max(int(b * (1 - sample_drop_ratio)), 1)
    brange = (paddle.randperm(b))[:sample_subset_size]
    x_subset = x[brange]

    # 2) apply residual_func to get residual
    residual = residual_func(x_subset)

    x_flat = x.flatten(1)
    residual = residual.flatten(1)

    residual_scale_factor = b / sample_subset_size

    # 3) add the residual
    x_plus_residual = paddle.index_add_(x_flat, 0, brange, residual, alpha=residual_scale_factor)
    return x_plus_residual.reshape(x.shape)


class SwiGLUFFN(nn.Layer):
    def __init__(
        self,
        in_features,
        hidden_features= None,
        out_features=None,
        act_layer=None,
        drop=0.0,
        bias=True):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.w12 = nn.Linear(in_features, 2 * hidden_features, bias_attr=bias)
        self.w3 = nn.Linear(hidden_features, out_features, bias_attr=bias)

    def forward(self, x):
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, axis=-1)
        hidden = F.silu(x1) * x2
        return self.w3(hidden)


class SwiGLUFFNFused(SwiGLUFFN):
    def __init__(
        self,
        in_features,
        hidden_features= None,
        out_features=None,
        act_layer=None,
        drop=0.0,
        bias=True):
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        hidden_features = (int(hidden_features * 2 / 3) + 7) // 8 * 8
        super().__init__(
            in_features=in_features,
            hidden_features=hidden_features,
            out_features=out_features,
            bias=bias,
        )


class PatchEmbed(nn.Layer):
    """
    2D image to patch embedding: (B,C,H,W) -> (B,N,D)

    Args:
        img_size: Image size.
        patch_size: Patch token size.
        in_chans: Number of input image channels.
        embed_dim: Number of linear projection output channels.
        norm_layer: Normalization layer.
    """

    def __init__(
        self,
        img_size=518, # to set up the position embeddings properly
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        norm_layer=None,
        flatten_embedding=True):
        super().__init__()
        image_HW = to_2tuple(img_size)
        patch_HW = to_2tuple(patch_size)
        patch_grid_size = (
            image_HW[0] // patch_HW[0],
            image_HW[1] // patch_HW[1],
        )

        self.img_size = image_HW
        self.patch_size = patch_HW
        self.patches_resolution = patch_grid_size
        self.num_patches = patch_grid_size[0] * patch_grid_size[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.flatten_embedding = flatten_embedding

        self.proj = nn.Conv2D(in_chans, embed_dim, kernel_size=patch_HW, stride=patch_HW)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        _, _, H, W = x.shape
        patch_H, patch_W = self.patch_size

        assert H % patch_H == 0, f"Input image height {H} is not a multiple of patch height {patch_H}"
        assert W % patch_W == 0, f"Input image width {W} is not a multiple of patch width: {patch_W}"

        x = self.proj(x)  # B C H W
        H, W = x.shape[2], x.shape[3] # [256, 384, 16, 16]
        x = x.flatten(2).transpose([0, 2, 1])  # B HW C
        x = self.norm(x)
        if not self.flatten_embedding:
            x = x.reshape([-1, H, W, self.embed_dim])  # B H W C
        return x


class BlockChunk(nn.LayerList):
    def forward(self, x):
        for b in self:
            x = b(x)
        return x


class DINOv2VisionTransformer(nn.Layer):
    """ DINOv2 Vision Transformer """

    def __init__(
        self,
        img_size=518, # to set up the position embeddings properly
        patch_size=14,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        ffn_bias=True,
        proj_bias=True,
        drop_path_rate=0.0,
        drop_path_uniform=False,
        init_values=1e-5, # None,  # for layerscale: None or 0 => no layerscale
        embed_layer=PatchEmbed,
        act_layer=nn.GELU,
        block_fn=Block,
        ffn_layer="mlp",
        block_chunks=0, #
        n_last_blocks=1,
        avgpool_patchtokens=True,
        **kwargs):
        super().__init__()
        norm_layer = partial(nn.LayerNorm, epsilon=1e-6)

        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 1
        self.n_blocks = depth
        self.num_heads = num_heads
        self.patch_size = patch_size

        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.pos_embed = self.create_parameter(
            shape=(1, num_patches + 1, embed_dim),
            default_initializer=paddle.nn.initializer.Constant(value=0.))
        self.cls_token = self.create_parameter(
            shape=(1, 1, embed_dim),
            default_initializer=paddle.nn.initializer.Constant(value=0.))

        if drop_path_uniform is True:
            dpr = [drop_path_rate] * depth
        else:
            dpr = np.linspace(0, drop_path_rate, depth)  # stochastic depth decay rule

        if ffn_layer == "mlp":
            logger.info("using MLP layer as FFN")
            ffn_layer = Mlp
        elif ffn_layer == "swiglufused" or ffn_layer == "swiglu":
            logger.info("using SwiGLU layer as FFN")
            ffn_layer = SwiGLUFFNFused
        elif ffn_layer == "identity":
            logger.info("using Identity layer as FFN")

            def f(*args, **kwargs):
                return nn.Identity()

            ffn_layer = f
        else:
            raise NotImplementedError

        blocks_list = [
            block_fn(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                ffn_bias=ffn_bias,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                ffn_layer=ffn_layer,
                init_values=init_values,
            )
            for i in range(depth)
        ]
        if block_chunks > 0:
            self.chunked_blocks = True
            chunked_blocks = []
            chunksize = depth // block_chunks
            for i in range(0, depth, chunksize):
                # this is to keep the block index consistent if we chunk the block list
                chunked_blocks.append([nn.Identity()] * i + blocks_list[i : i + chunksize])
            self.blocks = nn.LayerList([BlockChunk(p) for p in chunked_blocks])
        else:
            self.chunked_blocks = False
            self.blocks = nn.LayerList(blocks_list)

        self.norm = norm_layer(embed_dim)
        self.head = nn.Identity() # TODO

        self.mask_token = self.create_parameter(
            shape=(1, embed_dim),
            default_initializer=paddle.nn.initializer.Constant(value=0.))

        self.n_last_blocks = n_last_blocks
        self.avgpool_patchtokens = avgpool_patchtokens

        init.normal_(self.pos_embed, std=0.02)
        init.normal_(self.cls_token, std=1e-6)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.LayerNorm):
            init.zeros_(m.bias)
            init.ones_(m.weight)

    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        pos_embed = self.pos_embed
        class_pos_embed = pos_embed[:, 0]
        patch_pos_embed = pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1

        patch_pos_embed = F.interpolate(
            patch_pos_embed.reshape([1, int(math.sqrt(N)), int(math.sqrt(N)), dim]).transpose([0, 3, 1, 2]),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode="bicubic",
        )

        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.transpose([0, 2, 3, 1]).reshape([1, -1, dim])
        return paddle.concat((class_pos_embed.unsqueeze(0), patch_pos_embed), axis=1)

    def prepare_tokens_with_masks(self, x, masks=None):
        B, nc, w, h = x.shape # [256, 3, 224, 224]
        x = self.patch_embed(x) # [256, 256, 384]
        if masks is not None:
            x = paddle.where(masks.unsqueeze(-1), self.mask_token.unsqueeze(0), x)
        x = paddle.concat((self.cls_token.expand([x.shape[0], -1, -1]), x), axis=1)
        x = x + self.interpolate_pos_encoding(x, w, h)
        return x

    def forward_features_list(self, x_list, masks_list):
        x = [self.prepare_tokens_with_masks(x, masks) for x, masks in zip(x_list, masks_list)]
        for blk in self.blocks:
            x = blk(x)

        all_x = x
        output = []
        for x, masks in zip(all_x, masks_list):
            x_norm = self.norm(x)
            output.append(
                {
                    "x_norm_clstoken": x_norm[:, 0],
                    "x_norm_patchtokens": x_norm[:, 1:],
                    "x_prenorm": x,
                    "masks": masks,
                }
            )
        return output

    def forward_features(self, x, masks=None):
        if isinstance(x, list):
            return self.forward_features_list(x, masks)

        x = self.prepare_tokens_with_masks(x, masks)

        for blk in self.blocks:
            x = blk(x)

        x_norm = self.norm(x)
        return {
            "x_norm_clstoken": x_norm[:, 0],
            "x_norm_patchtokens": x_norm[:, 1:],
            "x_prenorm": x,
            "masks": masks,
        }

    def _get_intermediate_layers_not_chunked(self, x, n=1):
        x = self.prepare_tokens_with_masks(x)
        # If n is an int, take the n last blocks. If it's a list, take them
        output, total_block_len = [], len(self.blocks)
        blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in blocks_to_take:
                output.append(x)
        assert len(output) == len(blocks_to_take), f"only {len(output)} / {len(blocks_to_take)} blocks found"
        return output

    def _get_intermediate_layers_chunked(self, x, n=1):
        x = self.prepare_tokens_with_masks(x)
        output, i, total_block_len = [], 0, len(self.blocks[-1])
        # If n is an int, take the n last blocks. If it's a list, take them
        blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        for block_chunk in self.blocks:
            for blk in block_chunk[i:]:  # Passing the nn.Identity()
                x = blk(x)
                if i in blocks_to_take:
                    output.append(x)
                i += 1
        assert len(output) == len(blocks_to_take), f"only {len(output)} / {len(blocks_to_take)} blocks found"
        return output

    def get_intermediate_layers(self, x, n=1, reshape=False, return_class_token=False, norm=True):
        if self.chunked_blocks:
            outputs = self._get_intermediate_layers_chunked(x, n)
        else:
            outputs = self._get_intermediate_layers_not_chunked(x, n)
        if norm:
            outputs = [self.norm(out) for out in outputs]
        class_tokens = [out[:, 0] for out in outputs]
        outputs = [out[:, 1:] for out in outputs]
        if reshape:
            B, _, w, h = x.shape
            outputs = [
                out.reshape([B, w // self.patch_size, h // self.patch_size, -1]).transpose([0, 3, 1, 2])
                for out in outputs
            ]
        if return_class_token:
            return tuple(zip(outputs, class_tokens))
        return tuple(outputs)

    def forward(self, *args, **kwargs):
        ret = self.forward_features(*args, **kwargs)
        if self.training:
            return ret
        else:
            return self.head(ret["x_norm_clstoken"])


class DINOv2(Model):

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

            logger.info("There are {}/{} variables loaded into {} with {}.".format(
                num_params_loaded, len(model_state_dict), tag, path))
            self.set_dict(model_state_dict)
        else:
            logger.warning("No pretrained weights found in {}".format(path))

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


class DINOv2LinearProbe(DINOv2):

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
            x = self.backbone.forward_features(inp)
            cls_token = x["x_norm_clstoken"]
            patch_tokens = x["x_norm_patchtokens"]
            # fmt: off
            output = paddle.concat([
                cls_token,
                paddle.mean(patch_tokens, 1),
            ], 1)
        output = self.linear(output)
        return output


def dinov2_vit_small(patch_size=14, **kwargs):
    model = DINOv2VisionTransformer(
        patch_size=patch_size, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, epsilon=1e-6), **kwargs)
    return model


def dinov2_vit_small_linearprobe(patch_size=14, **kwargs):
    model = DINOv2LinearProbe(
        patch_size=patch_size,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(
            nn.LayerNorm, epsilon=1e-6),
        n_last_blocks=1,
        avgpool_patchtokens=True,
        **kwargs)
    return model


def dinov2_vit_base(patch_size=14, **kwargs):
    model = DINOv2VisionTransformer(
        patch_size=patch_size, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, epsilon=1e-6), **kwargs)
    return model


def dinov2_vit_base_linearprobe(patch_size=14, **kwargs):
    model = DINOv2LinearProbe(
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


def dinov2_vit_large(patch_size=14, **kwargs):
    model = DINOv2VisionTransformer(
        patch_size=patch_size, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, epsilon=1e-6), **kwargs)
    return model


def dinov2_vit_large_linearprobe(patch_size=14, **kwargs):
    model = DINOv2LinearProbe(
        patch_size=patch_size,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(
            nn.LayerNorm, epsilon=1e-6),
        n_last_blocks=1,
        avgpool_patchtokens=True,
        **kwargs)
    return model


def dinov2_vit_gaint2(patch_size=14, **kwargs):
    model = DINOv2VisionTransformer(
        patch_size=patch_size, embed_dim=1536, depth=40, num_heads=24, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, epsilon=1e-6), **kwargs)
    return model


def dinov2_vit_gaint2_linearprobe(patch_size=14, **kwargs):
    model = DINOv2LinearProbe(
        patch_size=patch_size,
        embed_dim=1536,
        depth=40,
        num_heads=24,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(
            nn.LayerNorm, epsilon=1e-6),
        n_last_blocks=1,
        avgpool_patchtokens=True,
        **kwargs)
    return model
