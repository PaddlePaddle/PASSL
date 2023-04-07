# copyright (c) 2023 PaddlePaddle Authors. All Rights Reserve.
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

# Code was based on:
# (1) https://github.com/rwightman/pytorch-image-models/blob/main/timm/models/swin_transformer.py
# (2) https://github.com/PaddlePaddle/PASSL/blob/main/passl/modeling/backbones/swin_transformer.py

from typing import Optional
from itertools import repeat
import collections.abc
import os
import math
import numpy as np

import paddle
import paddle.nn as nn
from .vision_transformer import Mlp, PatchEmbed, DropPath
from passl.nn import init
from passl.models.base_model import Model

__all__ = [
    'SwinTransformer',
    'swin_base_patch4_window12_384',
    'swin_base_patch4_window7_224',
    'swin_large_patch4_window12_384',
    'swin_large_patch4_window7_224',
    'swin_small_patch4_window7_224',
    'swin_tiny_patch4_window7_224',
    'swin_base_patch4_window12_384_in22k',
    'swin_base_patch4_window7_224_in22k',
    'swin_large_patch4_window12_384_in22k',
    'swin_large_patch4_window7_224_in22k',
    'swin_s3_tiny_224',
    'swin_s3_small_224',
    'swin_s3_base_224',
]


# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))

    return parse


to_2tuple = _ntuple(2)
to_ntuple = _ntuple


def window_partition(x, window_size: int):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.reshape(
        [B, H // window_size, window_size, W // window_size, window_size, C])
    windows = x.transpose([0, 1, 3, 2, 4, 5]).reshape(
        [-1, window_size, window_size, C])
    return windows


def window_reverse(windows, window_size: int, H: int, W: int):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.reshape(
        [B, H // window_size, W // window_size, window_size, window_size, -1])
    x = x.transpose([0, 1, 3, 2, 4, 5]).reshape([B, H, W, -1])
    return x


def get_relative_position_index(win_h, win_w):
    # get pair-wise relative position index for each token inside the window
    coords = paddle.stack(
        paddle.meshgrid(
            [paddle.arange(win_h), paddle.arange(win_w)]))  # 2, Wh, Ww
    coords_flatten = paddle.flatten(coords, 1)  # 2, Wh*Ww
    relative_coords = coords_flatten.unsqueeze(
        axis=2) - coords_flatten.unsqueeze(axis=1)  # 2, Wh*Ww, Wh*Ww
    relative_coords = relative_coords.transpose((1, 2, 0))  # Wh*Ww, Wh*Ww, 2
    relative_coords[:, :, 0] += win_h - 1  # shift to start from 0
    relative_coords[:, :, 1] += win_w - 1
    relative_coords[:, :, 0] *= 2 * win_w - 1
    return relative_coords.sum(-1)  # Wh*Ww, Wh*Ww


class WindowAttention(nn.Layer):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        head_dim (int): Number of channels per head (dim // num_heads if not set)
        window_size (tuple[int]): The height and width of the window.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self,
                 dim,
                 num_heads,
                 head_dim=None,
                 window_size=7,
                 qkv_bias=True,
                 attn_drop=0.,
                 proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = to_2tuple(window_size)  # Wh, Ww
        win_h, win_w = self.window_size
        self.window_area = win_h * win_w
        self.num_heads = num_heads
        head_dim = head_dim or dim // num_heads
        attn_dim = head_dim * num_heads
        self.scale = head_dim**-0.5

        # define a parameter table of relative position bias, shape: 2*Wh-1 * 2*Ww-1, nH
        self.relative_position_bias_table = self.create_parameter(shape=(
            (2 * win_h - 1) * (2 * win_w - 1), num_heads))
        init.zeros_(self.relative_position_bias_table)

        # get pair-wise relative position index for each token inside the window
        self.register_buffer("relative_position_index",
                             get_relative_position_index(win_h, win_w))

        self.qkv = nn.Linear(dim, attn_dim * 3, bias_attr=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(attn_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        init.trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(axis=-1)

    def _get_rel_pos_bias(self):
        index = self.relative_position_index.reshape([-1])
        relative_position_bias = paddle.index_select(
            self.relative_position_bias_table, index)
        relative_position_bias = relative_position_bias.reshape(
            [self.window_area, self.window_area, -1])  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.transpose(
            [2, 0, 1])  # nH, Wh*Ww, Wh*Ww
        return relative_position_bias.unsqueeze(0)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape((B_, N, 3, self.num_heads, -1)).transpose(
            (2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @k.transpose((0, 1, 3, 2)))
        attn = attn + self._get_rel_pos_bias()

        if mask is not None:
            num_win = mask.shape[0]
            attn = attn.reshape((B_ // num_win, num_win, self.num_heads, N, N
                                 )) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.reshape((-1, self.num_heads, N, N))
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @v).transpose((0, 2, 1, 3)).reshape((B_, N, -1))
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Layer):
    r""" Swin Transformer Block.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        window_size (int): Window size.
        num_heads (int): Number of attention heads.
        head_dim (int): Enforce the number of channels per head
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self,
                 dim,
                 input_resolution,
                 num_heads=4,
                 head_dim=None,
                 window_size=7,
                 shift_size=0,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim,
            num_heads=num_heads,
            head_dim=head_dim,
            window_size=to_2tuple(self.window_size),
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop)

        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim,
                       hidden_features=int(dim * mlp_ratio),
                       act_layer=act_layer,
                       drop=drop)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = paddle.zeros((1, H, W, 1))  # 1 H W 1
            cnt = 0
            for h in (slice(0, -self.window_size),
                      slice(-self.window_size, -self.shift_size),
                      slice(-self.shift_size, None)):
                for w in (slice(0, -self.window_size),
                          slice(-self.window_size, -self.shift_size),
                          slice(-self.shift_size, None)):
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            mask_windows = window_partition(
                img_mask,
                self.window_size)  # num_win, window_size, window_size, 1
            mask_windows = mask_windows.reshape(
                (-1, self.window_size * self.window_size))
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            huns = -100.0 * paddle.ones_like(attn_mask)
            attn_mask = huns * (attn_mask != 0).astype("float32")
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.reshape((B, H, W, C))

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = paddle.roll(
                x, shifts=(-self.shift_size, -self.shift_size), axis=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(
            shifted_x,
            self.window_size)  # num_win*B, window_size, window_size, C
        x_windows = x_windows.reshape(
            (-1, self.window_size * self.window_size,
             C))  # num_win*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(
            x_windows,
            mask=self.attn_mask)  # num_win*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.reshape(
            (-1, self.window_size, self.window_size, C))
        shifted_x = window_reverse(attn_windows, self.window_size, H,
                                   W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = paddle.roll(
                shifted_x,
                shifts=(self.shift_size, self.shift_size),
                axis=(1, 2))
        else:
            x = shifted_x
        x = x.reshape((B, H * W, C))

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class PatchMerging(nn.Layer):
    r""" Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self,
                 input_resolution,
                 dim,
                 out_dim=None,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.out_dim = out_dim or 2 * dim
        self.norm = norm_layer(4 * dim)
        self.reduction = nn.Linear(4 * dim, self.out_dim, bias_attr=False)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.reshape([B, H, W, C])

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = paddle.concat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.reshape((B, -1, 4 * C))  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x


class BasicLayer(nn.Layer):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        head_dim (int): Channels per head (dim // num_heads if not set)
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
    """

    def __init__(self,
                 dim,
                 out_dim,
                 input_resolution,
                 depth,
                 num_heads=4,
                 head_dim=None,
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth

        # build blocks
        self.blocks = nn.LayerList([
            SwinTransformerBlock(
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                head_dim=head_dim,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i]
                if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer) for i in range(depth)
        ])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(
                input_resolution,
                dim=dim,
                out_dim=out_dim,
                norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class SwinTransformer(Model):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030
    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        head_dim (int, tuple(int)):
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
    """

    def __init__(self,
                 img_size=224,
                 patch_size=4,
                 in_chans=3,
                 num_classes=1000,
                 global_pool='avg',
                 embed_dim=96,
                 depths=(2, 2, 6, 2),
                 num_heads=(3, 6, 12, 24),
                 head_dim=None,
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm,
                 ape=False,
                 patch_norm=True,
                 weight_init='',
                 **kwargs):
        super().__init__()
        assert global_pool in ('', 'avg')
        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.num_features = int(embed_dim * 2**(self.num_layers - 1))

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if patch_norm else None)
        num_patches = self.patch_embed.num_patches
        self.patch_grid = self.patch_embed.grid_size

        # absolute position embedding
        self.absolute_pos_embed = self.create_parameter(
            (1, num_patches, embed_dim)) if ape else None
        self.pos_drop = nn.Dropout(p=drop_rate)

        # build layers
        if not isinstance(embed_dim, (tuple, list)):
            embed_dim = [int(embed_dim * 2**i) for i in range(self.num_layers)]
        embed_out_dim = embed_dim[1:] + [None]
        head_dim = to_ntuple(self.num_layers)(head_dim)
        window_size = to_ntuple(self.num_layers)(window_size)
        mlp_ratio = to_ntuple(self.num_layers)(mlp_ratio)
        dpr = np.linspace(0, drop_path_rate,
                          sum(depths)).tolist()  # stochastic depth decay rule
        self.layers = nn.LayerList()
        for i in range(self.num_layers):
            self.layers.append(
                BasicLayer(
                    dim=embed_dim[i],
                    out_dim=embed_out_dim[i],
                    input_resolution=(self.patch_grid[0] // (2**i),
                                      self.patch_grid[1] // (2**i)),
                    depth=depths[i],
                    num_heads=num_heads[i],
                    head_dim=head_dim[i],
                    window_size=window_size[i],
                    mlp_ratio=mlp_ratio[i],
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                    norm_layer=norm_layer,
                    downsample=PatchMerging
                    if (i < self.num_layers - 1) else None))

        self.norm = norm_layer(self.num_features)
        self.head = nn.Linear(
            self.num_features,
            num_classes) if num_classes > 0 else nn.Identity()

        if self.absolute_pos_embed is not None:
            init.trunc_normal_(self.absolute_pos_embed, std=.02)

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
        nwd = {'absolute_pos_embed'}
        for n, _ in self.named_parameters():
            if 'relative_position_bias_table' in n:
                nwd.add(n)
        return nwd

    def group_matcher(self, coarse=False):
        return dict(
            stem=r'^absolute_pos_embed|patch_embed',  # stem and embed
            blocks=r'^layers\.(\d+)' if coarse else [
                (r'^layers\.(\d+).downsample', (0, )),
                (r'^layers\.(\d+)\.\w+\.(\d+)', None),
                (r'^norm', (99999, )),
            ])

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=None):
        self.num_classes = num_classes
        if global_pool is not None:
            assert global_pool in ('', 'avg')
            self.global_pool = global_pool
        self.head = nn.Linear(
            self.num_features,
            num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        x = self.patch_embed(x)
        if self.absolute_pos_embed is not None:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)  # B L C
        return x

    def forward_head(self, x, pre_logits: bool=False):
        if self.global_pool == 'avg':
            x = x.mean(axis=1)
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

        self.set_dict(param_state_dict)

    def save(self, path, local_rank=0, rank=0):
        paddle.save(self.state_dict(), path + ".pdparams")


def swin_base_patch4_window12_384(**kwargs):
    """ Swin-B @ 384x384, pretrained ImageNet-22k, fine tune 1k
    """
    model = SwinTransformer(
        img_size=384,
        patch_size=4,
        window_size=12,
        embed_dim=128,
        depths=(2, 2, 18, 2),
        num_heads=(4, 8, 16, 32),
        **kwargs)
    return model


def swin_base_patch4_window7_224(**kwargs):
    """ Swin-B @ 224x224, pretrained ImageNet-22k, fine tune 1k
    """
    model = SwinTransformer(
        img_size=224,
        patch_size=4,
        window_size=7,
        embed_dim=128,
        depths=(2, 2, 18, 2),
        num_heads=(4, 8, 16, 32),
        **kwargs)
    return model


def swin_large_patch4_window12_384(**kwargs):
    """ Swin-L @ 384x384, pretrained ImageNet-22k, fine tune 1k
    """
    model = SwinTransformer(
        img_size=384,
        patch_size=4,
        window_size=12,
        embed_dim=192,
        depths=(2, 2, 18, 2),
        num_heads=(6, 12, 24, 48),
        **kwargs)
    return model


def swin_large_patch4_window7_224(**kwargs):
    """ Swin-L @ 224x224, pretrained ImageNet-22k, fine tune 1k
    """
    model = SwinTransformer(
        img_size=224,
        patch_size=4,
        window_size=7,
        embed_dim=192,
        depths=(2, 2, 18, 2),
        num_heads=(6, 12, 24, 48),
        **kwargs)
    return model


def swin_small_patch4_window7_224(**kwargs):
    """ Swin-S @ 224x224, trained ImageNet-1k
    """
    model = SwinTransformer(
        img_size=224,
        patch_size=4,
        window_size=7,
        embed_dim=96,
        depths=(2, 2, 18, 2),
        num_heads=(3, 6, 12, 24),
        **kwargs)
    return model


def swin_tiny_patch4_window7_224(**kwargs):
    """ Swin-T @ 224x224, trained ImageNet-1k
    """
    model = SwinTransformer(
        img_size=224,
        patch_size=4,
        window_size=7,
        embed_dim=96,
        depths=(2, 2, 6, 2),
        num_heads=(3, 6, 12, 24),
        **kwargs)
    return model


def swin_base_patch4_window12_384_in22k(**kwargs):
    """ Swin-B @ 384x384, trained ImageNet-22k
    """
    model = SwinTransformer(
        img_size=384,
        patch_size=4,
        window_size=12,
        embed_dim=128,
        depths=(2, 2, 18, 2),
        num_heads=(4, 8, 16, 32),
        **kwargs)
    return model


def swin_base_patch4_window7_224_in22k(**kwargs):
    """ Swin-B @ 224x224, trained ImageNet-22k
    """
    model = SwinTransformer(
        img_size=224,
        patch_size=4,
        window_size=7,
        embed_dim=128,
        depths=(2, 2, 18, 2),
        num_heads=(4, 8, 16, 32),
        **kwargs)
    return model


def swin_large_patch4_window12_384_in22k(**kwargs):
    """ Swin-L @ 384x384, trained ImageNet-22k
    """
    model = SwinTransformer(
        img_size=384,
        patch_size=4,
        window_size=12,
        embed_dim=192,
        depths=(2, 2, 18, 2),
        num_heads=(6, 12, 24, 48),
        **kwargs)
    return model


def swin_large_patch4_window7_224_in22k(**kwargs):
    """ Swin-L @ 224x224, trained ImageNet-22k
    """
    model = SwinTransformer(
        img_size=224,
        patch_size=4,
        window_size=7,
        embed_dim=192,
        depths=(2, 2, 18, 2),
        num_heads=(6, 12, 24, 48),
        **kwargs)
    return model


def swin_s3_tiny_224(**kwargs):
    """ Swin-S3-T @ 224x224, ImageNet-1k. https://arxiv.org/abs/2111.14725
    """
    model = SwinTransformer(
        patch_size=4,
        window_size=(7, 7, 14, 7),
        embed_dim=96,
        depths=(2, 2, 6, 2),
        num_heads=(3, 6, 12, 24),
        **kwargs)
    return model


def swin_s3_small_224(**kwargs):
    """ Swin-S3-S @ 224x224, trained ImageNet-1k. https://arxiv.org/abs/2111.14725
    """
    model = SwinTransformer(
        patch_size=4,
        window_size=(14, 14, 14, 7),
        embed_dim=96,
        depths=(2, 2, 18, 2),
        num_heads=(3, 6, 12, 24),
        **kwargs)
    return model


def swin_s3_base_224(**kwargs):
    """ Swin-S3-B @ 224x224, trained ImageNet-1k. https://arxiv.org/abs/2111.14725
    """
    model = SwinTransformer(
        patch_size=4,
        window_size=(7, 7, 14, 7),
        embed_dim=96,
        depths=(2, 2, 30, 2),
        num_heads=(3, 6, 12, 24),
        **kwargs)
    return model
