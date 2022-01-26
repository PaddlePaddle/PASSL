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

# Code was based on https://github.com/facebookresearch/mae

from functools import partial
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from ...modules.get_sincos_pe import get_2d_sincos_pos_embed

from .builder import BACKBONES

trunc_normal_ = nn.initializer.TruncatedNormal(std=0.02)
xavier_uniform_ = nn.initializer.XavierUniform()
zeros_ = nn.initializer.Constant(value=0.0)
ones_ = nn.initializer.Constant(value=1.0)


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


class Identity(nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input


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
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class PatchEmbed(nn.Layer):
    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_chans=3,
                 embed_dim=768,
                 norm_layer=None,
                 flatten=True):
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
        assert H == self.img_size[
            0], f"Input image height ({H}) doesn't match model ({self.img_size[0]})."
        assert W == self.img_size[
            1], f"Input image width ({W}) doesn't match model ({self.img_size[1]})."
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose([0, 2, 1])  # BCHW -> BNC
        x = self.norm(x)
        return x


class Attention(nn.Layer):
    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 attn_drop=0.,
                 proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias_attr=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(
            [B, N, 3, self.num_heads,
             C // self.num_heads]).transpose([2, 0, 3, 1, 4])
        q, k, v = qkv.unbind(0)

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


class VisionTransformer(nn.Layer):
    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_chans=3,
                 num_classes=1000,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 embed_layer=PatchEmbed,
                 norm_layer=None,
                 act_layer=None,
                 weight_init=''):

        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        norm_layer = norm_layer or partial(nn.LayerNorm, epsilon=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(img_size=img_size,
                                       patch_size=patch_size,
                                       in_chans=in_chans,
                                       embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = paddle.create_parameter(
            shape=[1, 1, embed_dim],
            dtype='float32',
            default_initializer=trunc_normal_)

        self.pos_embed = paddle.create_parameter(
            shape=[1, num_patches + 1, embed_dim],
            dtype='float32',
            default_initializer=trunc_normal_)

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in paddle.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.Sequential(*[
            Block(dim=embed_dim,
                  num_heads=num_heads,
                  mlp_ratio=mlp_ratio,
                  qkv_bias=qkv_bias,
                  drop=drop_rate,
                  attn_drop=attn_drop_rate,
                  drop_path=dpr[i],
                  norm_layer=norm_layer,
                  act_layer=act_layer) for i in range(depth)
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

        x = self.patch_embed(x)

        cls_token = self.cls_token.expand([x.shape[0], -1, -1])

        x = paddle.concat([cls_token, x], axis=1)

        x = self.pos_drop(x + self.pos_embed)

        x = self.blocks(x)
        x = self.norm(x)
        return x[:, 0]

    def forward(self, x):
        x = self.forward_features(x)

        return x


@BACKBONES.register()
class MAE_ViT(VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, global_pool=True, **kwargs):
        super().__init__(**kwargs)

        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = partial(nn.LayerNorm, epsilon=1e-6)
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand([B, -1, -1])

        x = paddle.concat([cls_tokens, x], axis=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        if self.global_pool:
            x = x[:, 1:, :].mean(axis=1)
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        return outcome


@BACKBONES.register()
class MAE(nn.Layer):
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
                 norm_layer=partial(nn.LayerNorm, epsilon=1e-6),
                 norm_pix_loss=False):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = paddle.create_parameter(
            shape=[1, 1, embed_dim],
            dtype='float32',
            default_initializer=trunc_normal_)

        self.pos_embed = paddle.create_parameter(
            shape=[1, num_patches + 1, embed_dim],
            dtype='float32',
            default_initializer=trunc_normal_)

        self.pos_embed.stop_gradient = True  # fixed sin-cos embedding

        self.blocks = nn.LayerList([
            Block(embed_dim,
                  num_heads,
                  mlp_ratio,
                  qkv_bias=True,
                  norm_layer=norm_layer) for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim,
                                       decoder_embed_dim,
                                       bias_attr=True)

        self.mask_token = paddle.create_parameter(
            shape=[1, 1, decoder_embed_dim],
            dtype='float32',
            default_initializer=zeros_)

        self.decoder_pos_embed = paddle.create_parameter(
            shape=[1, num_patches + 1, decoder_embed_dim],
            dtype='float32',
            default_initializer=zeros_)

        self.decoder_pos_embed.stop_gradient = True  # fixed sin-cos embedding

        self.decoder_blocks = nn.LayerList([
            Block(decoder_embed_dim,
                  decoder_num_heads,
                  mlp_ratio,
                  qkv_bias=True,
                  norm_layer=norm_layer) for i in range(decoder_depth)
        ])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim,
                                      patch_size**2 * in_chans,
                                      bias_attr=True)  # decoder to patch
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            int(self.patch_embed.num_patches**.5),
            cls_token=True)
        self.pos_embed.set_value(
            paddle.to_tensor(pos_embed).astype('float32').unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1],
            int(self.patch_embed.num_patches**.5),
            cls_token=True)
        self.decoder_pos_embed.set_value(
            paddle.to_tensor(decoder_pos_embed).astype('float32').unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight
        xavier_uniform_(w.reshape([w.shape[0], -1]))

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            zeros_(m.bias)
            ones_(m.weight)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape([imgs.shape[0], 3, h, p, w, p])
        x = paddle.einsum('nchpwq->nhwpqc', x)
        x = x.reshape([imgs.shape[0], h * w, p**2 * 3])
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]

        x = x.reshape([x.shape[0], h, w, p, p, 3])
        x = paddle.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape([x.shape[0], 3, h * p, h * p])
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = paddle.rand([N, L])  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = paddle.argsort(
            noise, axis=1)  # ascend: small is keep, large is remove
        ids_restore = paddle.argsort(ids_shuffle, axis=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]

        x_masked = x[paddle.arange(N)[:, None], ids_keep]

        # generate the binary mask: 0 is keep, 1 is remove
        mask = paddle.ones([N, L])
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = mask[paddle.arange(N)[:, None], ids_restore]

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand([x.shape[0], -1, -1])
        x = paddle.concat([cls_tokens, x], axis=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.tile(
            [x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1])
        x_ = paddle.concat([x[:, 1:, :], mask_tokens], axis=1)  # no cls token

        x_ = x_[paddle.arange(x.shape[0])[:, None], ids_restore]  # unshuffle

        x = paddle.concat([x[:, :1, :], x_], axis=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

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
