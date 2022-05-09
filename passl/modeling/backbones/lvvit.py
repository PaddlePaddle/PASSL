import numpy as np

import paddle
import paddle.nn as nn

from .builder import BACKBONES

__all__ = ["LVViT"]

trunc_normal_ = nn.initializer.TruncatedNormal(std=0.02)
zeros_ = nn.initializer.Constant(value=0.0)
ones_ = nn.initializer.Constant(value=1.0)


def to_2tuple(x):
    return tuple([x] * 2)


def drop_path(x, drop_prob=0., training=False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ...
    """
    if drop_prob == 0. or not training:
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
        # B= paddle.shape(x)[0]
        N, C = x.shape[1:]
        qkv = self.qkv(x).reshape(
            [-1, N, 3, self.num_heads, C // self.num_heads]).transpose(
                [2, 0, 3, 1, 4])
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q * self.scale) @k.transpose([0, 1, 3, 2])
        attn = nn.functional.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)

        x = (attn @v).transpose([0, 2, 1, 3]).reshape([-1, N, C])
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
                 norm_layer=nn.LayerNorm,
                 skip_lam=1.):
        super().__init__()
        self.skip_lam = skip_lam

        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity(
        )
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim,
                       hidden_features=mlp_hidden_dim,
                       act_layer=act_layer,
                       drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x))) / self.skip_lam
        x = x + self.drop_path(self.mlp(self.norm2(x))) / self.skip_lam
        return x


class PatchEmbedNaive(nn.Layer):
    """ 
    Image to Patch Embedding
    from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * \
            (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.embed_dim = embed_dim

        self.proj = nn.Conv2D(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        return x


class PatchEmbed4_2(nn.Layer):
    """ 
    Image to Patch Embedding with 4 layer convolution
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()

        new_patch_size = to_2tuple(patch_size // 2)

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * \
            (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.embed_dim = embed_dim

        self.conv1 = nn.Conv2D(
            in_chans, 64, 7, stride=2, padding=3, bias_attr=False)  # 112x112
        self.bn1 = nn.BatchNorm2D(64)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2D(
            64, 64, 3, stride=1, padding=1, bias_attr=False)  # 112x112
        self.bn2 = nn.BatchNorm2D(64)
        self.conv3 = nn.Conv2D(64, 64, 3, stride=1, padding=1, bias_attr=False)
        self.bn3 = nn.BatchNorm2D(64)

        self.proj = nn.Conv2D(
            64, embed_dim, new_patch_size, stride=new_patch_size)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.proj(x)  # [B, C, W, H]

        return x


class PatchEmbed4_2_128(nn.Layer):
    """ 
    Image to Patch Embedding with 4 layer convolution and 128 filters
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()

        new_patch_size = to_2tuple(patch_size // 2)

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * \
            (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.embed_dim = embed_dim

        self.conv1 = nn.Conv2D(
            in_chans, 128, kernel_size=7, stride=2, padding=3,
            bias_attr=False)  # 112x112
        self.bn1 = nn.BatchNorm2D(128)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2D(
            128, 128, kernel_size=3, stride=1, padding=1,
            bias_attr=False)  # 112x112
        self.bn2 = nn.BatchNorm2D(128)
        self.conv3 = nn.Conv2D(
            128, 128, kernel_size=3, stride=1, padding=1, bias_attr=False)
        self.bn3 = nn.BatchNorm2D(128)

        self.proj = nn.Conv2D(
            128, embed_dim, kernel_size=new_patch_size, stride=new_patch_size)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.proj(x)  # [B, C, W, H]

        return x


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


@BACKBONES.register()
class LVViT(nn.Layer):
    """ Vision Transformer with tricks
    Arguements:
        p_emb: different conv based position embedding (default: 4 layer conv)
        skip_lam: residual scalar for skip connection (default: 1.0)
        order: which order of layers will be used (default: None, will override depth if given)
        mix_token: use mix token augmentation for batch of tokens (default: False)
        return_dense: whether to return feature of all tokens with an additional aux_head (default: False)
    """

    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_chans=3,
                 class_num=1000,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_layer=nn.LayerNorm,
                 p_emb='pe4_2',
                 skip_lam=1.0,
                 mix_token=False,
                 return_dense=False):
        super().__init__()
        self.class_num = class_num
        # num_features for consistency with other models
        self.num_features = self.embed_dim = embed_dim
        self.output_dim = embed_dim if class_num == 0 else class_num

        if p_emb == 'pe4_2':
            patch_embed_fn = PatchEmbed4_2
        elif p_emb == 'pe4_2_128':
            patch_embed_fn = PatchEmbed4_2_128
        else:
            patch_embed_fn = PatchEmbedNaive

        self.patch_embed = patch_embed_fn(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim)

        num_patches = self.patch_embed.num_patches

        self.cls_token = self.create_parameter(
            shape=[1, 1, embed_dim], default_initializer=zeros_)
        self.pos_embed = self.create_parameter(
            shape=[1, num_patches + 1, embed_dim], default_initializer=zeros_)
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in paddle.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.LayerList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                skip_lam=skip_lam) for i in range(depth)
        ])

        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim,
                              class_num) if class_num > 0 else nn.Identity()

        self.return_dense = return_dense
        self.mix_token = mix_token

        if return_dense:
            self.aux_head = nn.Linear(
                embed_dim, class_num) if class_num > 0 else nn.Identity()
        if mix_token:
            self.beta = 1.0
            assert return_dense, "always return all features when mixtoken is enabled"

        trunc_normal_(self.pos_embed)
        trunc_normal_(self.cls_token)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            zeros_(m.bias)
            ones_(m.weight)

    def forward_embeddings(self, x):
        x = self.patch_embed(x)
        return x

    def forward_tokens(self, x):
        B = x.shape[0]
        cls_tokens = self.cls_token.expand([B, -1, -1])
        x = paddle.concat([cls_tokens, x], axis=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x

    def forward_features(self, x):
        # simple forward to obtain feature map (without mixtoken)
        x = self.forward_embeddings(x)
        x = x.flatten(2).transpose([0, 2, 1])
        x = self.forward_tokens(x)
        return x

    def forward(self, x):
        x = self.forward_embeddings(x)

        # token level mixtoken augmentation
        if self.mix_token and self.training:
            lam = np.random.beta(self.beta, self.beta)
            patch_h, patch_w = x.shape[2], x.shape[3]
            bbx1, bby1, bbx2, bby2 = rand_bbox(x.shape, lam)
            do_mixed = bbx1 < bbx2 and bbx2 < bby2
            if do_mixed:
                temp_x = x.clone()
                temp_x[:, :, bbx1:bbx2, bby1:bby2] = \
                    x.flip(0)[:, :, bbx1:bbx2, bby1:bby2]
                x = temp_x
        else:
            do_mixed = False
            bbx1, bby1, bbx2, bby2 = 0, 0, 0, 0

        x = x.flatten(2).transpose([0, 2, 1])
        x = self.forward_tokens(x)
        x_cls = self.head(x[:, 0])

        if self.return_dense:
            x_aux = self.aux_head(x[:, 1:])
            if not self.training:
                return x_cls + 0.5 * x_aux.max(1)[0]

            # recover the mixed part
            if do_mixed and self.mix_token and self.training:
                x_aux = x_aux.reshape(
                    [x_aux.shape[0], patch_h, patch_w, x_aux.shape[-1]])
                temp_x = x_aux.clone()
                temp_x[:, bbx1:bbx2, bby1:bby2, :] = \
                    x_aux.flip(0)[:, bbx1:bbx2, bby1:bby2, :]
                x_aux = temp_x
                x_aux = x_aux.reshape(
                    [x_aux.shape[0], patch_h * patch_w, x_aux.shape[-1]])

            return x_cls, x_aux, (bbx1, bby1, bbx2, bby2)
        return x_cls
