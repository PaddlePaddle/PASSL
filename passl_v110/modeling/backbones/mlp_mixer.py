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

# Code was based on https://github.com/lmk123568/pytorch-image-models/blob/master/timm/models/mlp_mixer.py

from functools import partial

import paddle
import paddle.nn as nn
from .builder import BACKBONES

normal_ = nn.initializer.Normal(std=1e-6)
zeros_ = nn.initializer.Constant(value=0.0)
ones_ = nn.initializer.Constant(value=1.0)
xavier_uniform_ = nn.initializer.XavierUniform()
trunc_normal_ = nn.initializer.TruncatedNormal(std=0.02)


class Identity(nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


def drop_path(x, drop_prob=0.0, training=False):

    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0], ) + (1, ) * (x.ndim - 1)
    random_tensor = paddle.to_tensor(keep_prob) + paddle.rand(shape)
    random_tensor = paddle.floor(random_tensor)
    output = x.divide(keep_prob) * random_tensor
    return output


class DropPath(nn.Layer):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class PatchEmbed(nn.Layer):
    """ 2D Image to Patch Embedding
    """
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        norm_layer=None,
        flatten=True,
    ):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0],
                          img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2D(in_chans,
                              embed_dim,
                              kernel_size=patch_size,
                              stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert (
            H == self.img_size[0]
        ), f"Input image height ({H}) doesn't match model ({self.img_size[0]})."
        assert (
            W == self.img_size[1]
        ), f"Input image width ({W}) doesn't match model ({self.img_size[1]})."
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose([0, 2, 1])  # BCHW -> BNC
        x = self.norm(x)
        return x


class Mlp(nn.Layer):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
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
        drop_probs = (drop, drop)

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class MixerBlock(nn.Layer):
    """ Residual Block w/ token mixing and channel MLPs
    Based on: 'MLP-Mixer: An all-MLP Architecture for Vision' - https://arxiv.org/abs/2105.01601
    """
    def __init__(
        self,
        dim,
        seq_len,
        mlp_ratio=(0.5, 4.0),
        mlp_layer=Mlp,
        norm_layer=partial(nn.LayerNorm, epsilon=1e-6),
        act_layer=nn.GELU,
        drop=0.0,
        drop_path=0.0,
    ):
        super().__init__()
        tokens_dim = int(mlp_ratio[0] * dim)
        channels_dim = int(mlp_ratio[1] * dim)
        self.norm1 = norm_layer(dim)
        self.mlp_tokens = mlp_layer(seq_len,
                                    tokens_dim,
                                    act_layer=act_layer,
                                    drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else Identity()
        self.norm2 = norm_layer(dim)
        self.mlp_channels = mlp_layer(dim,
                                      channels_dim,
                                      act_layer=act_layer,
                                      drop=drop)

    def forward(self, x):
        x = x + self.drop_path(
            self.mlp_tokens(self.norm1(x).transpose([0, 2, 1])).transpose(
                [0, 2, 1]))
        x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
        return x


@BACKBONES.register()
class MlpMixer(nn.Layer):
    def __init__(
        self,
        num_classes=1000,
        img_size=224,
        in_chans=3,
        patch_size=16,
        num_blocks=8,
        embed_dim=512,
        mlp_ratio=(0.5, 4.0),
        block_layer=MixerBlock,
        mlp_layer=Mlp,
        norm_layer=partial(nn.LayerNorm, epsilon=1e-6),
        act_layer=nn.GELU,
        drop_rate=0.0,
        drop_path_rate=0.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = (
            self.embed_dim
        ) = embed_dim  # num_features for consistency with other models

        self.stem = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        # FIXME drop_path (stochastic depth scaling rule or all the same?)
        self.blocks = nn.Sequential(*[
            block_layer(
                embed_dim,
                self.stem.num_patches,
                mlp_ratio,
                mlp_layer=mlp_layer,
                norm_layer=norm_layer,
                act_layer=act_layer,
                drop=drop_rate,
                drop_path=drop_path_rate,
            ) for _ in range(num_blocks)
        ])
        self.norm = norm_layer(embed_dim)

        self.init_weights()

    def init_weights(self):
        for n, m in self.named_sublayers():
            _init_weights(m, n)

    def forward_features(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.norm(x)
        x = x.mean(axis=1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        return x


def _init_weights(m, n):
    """ Mixer weight initialization
    """
    if isinstance(m, nn.Linear):
        if n.startswith("head"):
            zeros_(m.weight)
            zeros_(m.bias)
        else:
            xavier_uniform_(m.weight)
            if m.bias is not None:
                if "mlp" in n:
                    normal_(m.bias)
                else:
                    zeros_(m.bias)
    elif isinstance(m, nn.Conv2D):
        trunc_normal_(m.weight)
        if m.bias is not None:
            zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        zeros_(m.bias)
        ones_(m.weight)
