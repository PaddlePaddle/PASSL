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

import paddle
import paddle.nn as nn

from passl.models.base_model import Model
from passl.nn import init

from .conv_vit import CPatchEmbed, CBlock
from ..vision_transformer import Block
from ..utils.pos_embed import get_2d_sincos_pos_embed

__all__ = ['MaskedAutoencoderConvViT', 'convmae_convvit_base_patch16']


class MaskedAutoencoderConvViT(Model):
    """ Masked Autoencoder with VisionTransformer backbone
    """

    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_chans=3,
                 embed_dim=1024,
                 depth=24,
                 num_heads=16,
                 decoder_embed_dim=512,
                 decoder_depth=8,
                 decoder_num_heads=16,
                 mlp_ratio=4.,
                 norm_layer=nn.LayerNorm,
                 norm_pix_loss=False):
        super().__init__()
        # --------------------------------------------------------------------------
        # ConvMAE encoder specifics
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

        self.patch_embed4 = nn.Linear(embed_dim[2], embed_dim[2])
        self.stage1_output_decode = nn.Conv2D(
            embed_dim[0], embed_dim[2], 4, stride=4)
        self.stage2_output_decode = nn.Conv2D(
            embed_dim[1], embed_dim[2], 2, stride=2)

        num_patches = self.patch_embed3.num_patches
        self.pos_embed = self.create_parameter(shape=(1, num_patches,
                                                      embed_dim[2]))
        self.blocks1 = nn.LayerList([
            CBlock(
                dim=embed_dim[0],
                num_heads=num_heads,
                mlp_ratio=mlp_ratio[0],
                qkv_bias=True,
                qk_scale=None,
                norm_layer=norm_layer) for i in range(depth[0])
        ])
        self.blocks2 = nn.LayerList([
            CBlock(
                dim=embed_dim[1],
                num_heads=num_heads,
                mlp_ratio=mlp_ratio[1],
                qkv_bias=True,
                qk_scale=None,
                norm_layer=norm_layer) for i in range(depth[1])
        ])
        self.blocks3 = nn.LayerList([
            Block(
                dim=embed_dim[2],
                num_heads=num_heads,
                mlp_ratio=mlp_ratio[2],
                qkv_bias=True,
                qk_scale=None,
                norm_layer=norm_layer) for i in range(depth[2])
        ])
        self.norm = norm_layer(embed_dim[-1])

        # --------------------------------------------------------------------------
        # ConvMAE decoder specifics
        self.decoder_embed = nn.Linear(
            embed_dim[-1], decoder_embed_dim, bias_attr=True)

        self.mask_token = self.create_parameter(shape=(1, 1,
                                                       decoder_embed_dim))

        self.decoder_pos_embed = self.create_parameter(shape=(
            1, num_patches, decoder_embed_dim))  # fixed sin-cos embedding
        self.decoder_pos_embed.stop_gradient = True
        self.decoder_blocks = nn.LayerList([
            Block(
                decoder_embed_dim,
                decoder_num_heads,
                mlp_ratio[0],
                qkv_bias=True,
                qk_scale=None,
                norm_layer=norm_layer) for i in range(decoder_depth)
        ])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(
            decoder_embed_dim, (patch_size[0] * patch_size[1] * patch_size[2])
            **2 * in_chans,
            bias_attr=True)  # decoder to patch
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            int(self.patch_embed3.num_patches**.5),
            cls_token=False)
        self.pos_embed.copy_(
            paddle.to_tensor(pos_embed).astype(paddle.float32).unsqueeze(0),
            False)

        decoder_pos_embed = get_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1],
            int(self.patch_embed3.num_patches**.5),
            cls_token=False)
        self.decoder_pos_embed.copy_(
            paddle.to_tensor(decoder_pos_embed).astype(paddle.float32)
            .unsqueeze(0), False)

        # initialize patch_embed like nn.Linear (instead of nn.Conv2D)
        w = self.patch_embed3.proj.weight.reshape(
            [self.patch_embed3.proj.weight.shape[0], -1])
        init.xavier_uniform_(w)
        w._share_buffer_to(self.patch_embed3.proj.weight)

        init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            init.constant_(m.bias, 0)
            init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = 16
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = paddle.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = paddle.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N = x.shape[0]
        L = self.patch_embed3.num_patches
        #        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = paddle.rand([N, L])  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = paddle.argsort(
            noise, axis=1)  # ascend: small is keep, large is remove
        ids_restore = paddle.argsort(ids_shuffle, axis=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        # x_masked = paddle.gather(x, axis=1, index=ids_keep.unsqueeze(-1).tile((1, 1, D)))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = paddle.ones([N, L])
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        #mask = paddle.gather(mask, axis=1, index=ids_restore)
        mask = mask[paddle.arange(N).unsqueeze(1), ids_restore]

        return ids_keep, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):
        # embed patches
        ids_keep, mask, ids_restore = self.random_masking(x, mask_ratio)
        mask_for_patch1 = mask.reshape((-1, 14, 14)).unsqueeze(-1).tile(
            (1, 1, 1, 16)).reshape((-1, 14, 14, 4, 4)).transpose(
                (0, 1, 3, 2, 4)).reshape((x.shape[0], 56, 56)).unsqueeze(1)
        mask_for_patch2 = mask.reshape((-1, 14, 14)).unsqueeze(-1).tile(
            (1, 1, 1, 4)).reshape((-1, 14, 14, 2, 2)).transpose(
                (0, 1, 3, 2, 4)).reshape((x.shape[0], 28, 28)).unsqueeze(1)
        x = self.patch_embed1(x)
        for blk in self.blocks1:
            x = blk(x, 1 - mask_for_patch1)
        stage1_embed = self.stage1_output_decode(x).flatten(2).transpose(
            (0, 2, 1))

        x = self.patch_embed2(x)
        for blk in self.blocks2:
            x = blk(x, 1 - mask_for_patch2)
        stage2_embed = self.stage2_output_decode(x).flatten(2).transpose(
            (0, 2, 1))
        x = self.patch_embed3(x)
        x = x.flatten(2).transpose((0, 2, 1))
        x = self.patch_embed4(x)
        # add pos embed w/o cls token
        x = x + self.pos_embed
        x = x[paddle.arange(x.shape[0]).unsqueeze(1), ids_keep]
        stage1_embed = stage1_embed[paddle.arange(stage1_embed.shape[0])
                                    .unsqueeze(1), ids_keep]
        stage2_embed = stage2_embed[paddle.arange(stage2_embed.shape[0])
                                    .unsqueeze(1), ids_keep]

        # apply Transformer blocks
        for blk in self.blocks3:
            x = blk(x)
        x = x + stage1_embed + stage2_embed
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.tile(
            (x.shape[0], ids_restore.shape[1] - x.shape[1], 1))
        x_ = paddle.concat([x, mask_tokens], axis=1)  # no cls token
        x = x_[paddle.arange(x_.shape[0]).unsqueeze(1),
               ids_restore]  # unshuffle

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        return x

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(axis=-1, keepdim=True)
            var = target.var(axis=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target)**2
        loss = loss.mean(axis=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, imgs, mask_ratio=0.75):
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask


def convmae_convvit_base_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderConvViT(
        img_size=[224, 56, 28],
        patch_size=[4, 2, 2],
        embed_dim=[256, 384, 768],
        depth=[2, 2, 11],
        num_heads=12,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=[4, 4, 4],
        norm_layer=partial(
            nn.LayerNorm, epsilon=1e-6),
        **kwargs)
    return model


# set recommended archs
convmae_convvit_base_patch16 = convmae_convvit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
