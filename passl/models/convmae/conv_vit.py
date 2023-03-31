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

# Code was heavily based on https://github.com/Alpha-VL/ConvMAE/models_convvit.py

from functools import partial
import numpy as np

import paddle
import paddle.nn as nn

from passl.models.base_model import Model
from passl.nn import init

from ..vision_transformer import to_2tuple, DropPath, Block

__all__ = ['ConvViT', 'convvit_base_patch16']


class CMlp(nn.Layer):
    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2D(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2D(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class CBlock(nn.Layer):
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
        self.norm1 = nn.LayerNorm(dim)
        self.conv1 = nn.Conv2D(dim, dim, 1)
        self.conv2 = nn.Conv2D(dim, dim, 1)
        self.attn = nn.Conv2D(dim, dim, 5, padding=2, groups=dim)
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = CMlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop)

    def forward(self, x, mask=None):
        if mask is not None:
            x = x + self.drop_path(
                self.conv2(
                    self.attn(mask * self.conv1(
                        self.norm1(x.transpose((0, 2, 3, 1))).transpose((
                            0, 3, 1, 2))))))
        else:
            x = x + self.drop_path(
                self.conv2(
                    self.attn(
                        self.conv1(
                            self.norm1(x.transpose((0, 2, 3, 1))).transpose((
                                0, 3, 1, 2))))))
        x = x + self.drop_path(
            self.mlp(
                self.norm2(x.transpose((0, 2, 3, 1))).transpose((0, 3, 1, 2))))
        return x


class CPatchEmbed(nn.Layer):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.grid_size = (img_size[0] // patch_size[0],
                          img_size[1] // patch_size[1])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2D(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

        self.norm = nn.LayerNorm(embed_dim)
        self.act = nn.GELU()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        x = self.proj(x)
        x = self.norm(x.transpose((0, 2, 3, 1))).transpose((0, 3, 1, 2))
        x = self.act(x)
        return x


class HybridEmbed(nn.Layer):
    """ CNN Feature Map Embedding
    Extract feature map from CNN, flatten, project to embedding dim.
    """

    def __init__(self,
                 backbone,
                 img_size=224,
                 feature_size=None,
                 in_chans=3,
                 embed_dim=768):
        super().__init__()
        assert isinstance(backbone, nn.Layer)
        img_size = to_2tuple(img_size)
        self.img_size = img_size
        self.backbone = backbone
        if feature_size is None:
            with paddle.no_grad():
                # FIXME this is hacky, but most reliable way of determining the exact dim of the output feature
                # map for all networks, the feature metadata has reliable channel and stride info, but using
                # stride to calc feature dim requires info about padding of each stage that isn't captured.
                training = backbone.training
                if training:
                    backbone.eval()
                o = self.backbone(
                    paddle.zeros((1, in_chans, img_size[0], img_size[1])))[-1]
                feature_size = o.shape[-2:]
                feature_dim = o.shape[1]
                backbone.train(training)
        else:
            feature_size = to_2tuple(feature_size)
            feature_dim = self.backbone.feature_info.channels()[-1]
        self.num_patches = feature_size[0] * feature_size[1]
        self.proj = nn.Linear(feature_dim, embed_dim)

    def forward(self, x):
        x = self.backbone(x)[-1]
        x = x.flatten(2).transpose((1, 2))
        x = self.proj(x)
        return x


class ConvViT(Model):
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
                 hybrid_backbone=None,
                 norm_layer=nn.LayerNorm,
                 global_pool=False,
                 **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        if hybrid_backbone is not None:
            self.patch_embed = HybridEmbed(
                hybrid_backbone,
                img_size=img_size,
                in_chans=in_chans,
                embed_dim=embed_dim)
        else:
            self.patch_embed1 = CPatchEmbed(
                img_size=img_size[0],
                patch_size=patch_size[0],
                in_chans=in_chans,
                embed_dim=embed_dim[0])
            self.patch_embed2 = CPatchEmbed(
                img_size=img_size[1],
                patch_size=patch_size[1],
                in_chans=embed_dim[0],
                embed_dim=embed_dim[1])
            self.patch_embed3 = CPatchEmbed(
                img_size=img_size[2],
                patch_size=patch_size[2],
                in_chans=embed_dim[1],
                embed_dim=embed_dim[2])
        num_patches = self.patch_embed3.num_patches
        self.patch_embed4 = nn.Linear(embed_dim[2], embed_dim[2])
        self.pos_embed = self.create_parameter(shape=(1, num_patches,
                                                      embed_dim[2]))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = np.linspace(0, drop_path_rate,
                          sum(depth))  # stochastic depth decay rule
        self.blocks1 = nn.LayerList([
            CBlock(
                dim=embed_dim[0],
                num_heads=num_heads,
                mlp_ratio=mlp_ratio[0],
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer) for i in range(depth[0])
        ])
        self.blocks2 = nn.LayerList([
            CBlock(
                dim=embed_dim[1],
                num_heads=num_heads,
                mlp_ratio=mlp_ratio[1],
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[depth[0] + i],
                norm_layer=norm_layer) for i in range(depth[1])
        ])
        self.blocks3 = nn.LayerList([
            Block(
                dim=embed_dim[2],
                num_heads=num_heads,
                mlp_ratio=mlp_ratio[2],
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[depth[0] + depth[1] + i],
                norm_layer=norm_layer) for i in range(depth[2])
        ])

        self.norm = norm_layer(embed_dim[-1])

        # Classifier head
        self.head = nn.Linear(
            embed_dim[-1], num_classes) if num_classes > 0 else nn.Identity()

        init.trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)

        self.global_pool = global_pool
        if self.global_pool:
            self.fc_norm = norm_layer(embed_dim[-1])
            del self.norm  # remove the original norm

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            init.constant_(m.bias, 0)
            init.constant_(m.weight, 1.0)

    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(
            self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed1(x)
        x = self.pos_drop(x)
        for blk in self.blocks1:
            x = blk(x)
        x = self.patch_embed2(x)
        for blk in self.blocks2:
            x = blk(x)
        x = self.patch_embed3(x)
        x = x.flatten(2).transpose((0, 2, 1))
        x = self.patch_embed4(x)
        x = x + self.pos_embed
        for blk in self.blocks3:
            x = blk(x)
        if self.global_pool:
            x = x.mean(axis=1)
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]
        return outcome

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


def convvit_base_patch16(**kwargs):
    model = ConvViT(
        img_size=[224, 56, 28],
        patch_size=[4, 2, 2],
        embed_dim=[256, 384, 768],
        depth=[2, 2, 11],
        num_heads=12,
        mlp_ratio=[4, 4, 4],
        qkv_bias=True,
        norm_layer=partial(
            nn.LayerNorm, epsilon=1e-6),
        **kwargs)
    return model
