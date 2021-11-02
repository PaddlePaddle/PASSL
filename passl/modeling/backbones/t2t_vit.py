import copy
import math
import numpy as np
import paddle
import paddle.nn as nn

from .builder import BACKBONES

trunc_normal_ = nn.initializer.TruncatedNormal(std=0.02)
zeros_ = nn.initializer.Constant(value=0.0)
ones_ = nn.initializer.Constant(value=1.0)


def orthogonal(t, gain=1.):
    if t.ndim < 2:
        raise ValueError("Only tensors with 2 or more dimensions are supported")

    gain = paddle.to_tensor(gain)
    rows = t.shape[0]
    cols = t.numel() // rows
    flattened = paddle.normal(0, 1, [rows, cols])

    if rows < cols:
        flattened = flattened.transpose([1, 0])

    # Compute the qr factorization
    q, r = np.linalg.qr(flattened.cpu().numpy())
    q = paddle.to_tensor(q)
    r = paddle.to_tensor(r)
    d = paddle.diag(r, 0)
    ph = d.sign()
    q *= ph

    if rows < cols:
        q = q.transpose([1, 0])

    with paddle.no_grad():
        t = q
        #t.view_as(q).copy_(q)
        t = t.multiply(gain)
    return t


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
                 token_type='performer',
                 in_channels=3,
                 embed_dim=768,
                 token_dim=64):
        super().__init__()
        if token_type == 'transformer':

            self.attn1 = TokenTransformer(dim=in_channels * 7 * 7,
                                          in_dim=token_dim,
                                          num_heads=1,
                                          mlp_ratio=1.0)
            self.attn2 = TokenTransformer(dim=token_dim * 3 * 3,
                                          in_dim=token_dim,
                                          num_heads=1,
                                          mlp_ratio=1.0)

            w_attr_1, b_attr_1 = self._init_weights()
            self.proj = nn.Linear(token_dim * 3 * 3,
                                  embed_dim,
                                  weight_attr=w_attr_1,
                                  bias_attr=b_attr_1)

        elif token_type == 'performer':
            self.attn1 = TokenPerformer(dim=in_channels * 7 * 7,
                                        in_dim=token_dim,
                                        kernel_ratio=0.5)
            self.attn2 = TokenPerformer(dim=token_dim * 3 * 3,
                                        in_dim=token_dim,
                                        kernel_ratio=0.5)

            w_attr_1, b_attr_1 = self._init_weights()
            self.proj = nn.Linear(token_dim * 3 * 3,
                                  embed_dim,
                                  weight_attr=w_attr_1,
                                  bias_attr=b_attr_1)

        elif token_type == 'convolution':
            # 1st conv
            self.soft_split0 = nn.Conv2D(in_channels=in_channels,
                                         out_channels=token_dim,
                                         kernel_size=7,
                                         stride=4,
                                         padding=2)
            # 2nd conv
            self.soft_split1 = nn.Conv2D(in_channels=token_dim,
                                         out_channels=token_dim,
                                         kernel_size=3,
                                         stride=2,
                                         padding=1)
            # 3rd conv
            self.proj = nn.Conv2D(in_channels=token_dim,
                                  out_channels=embed_dim,
                                  kernel_size=3,
                                  stride=2,
                                  padding=1)
        else:
            raise ValueError(f'token_type: {token_type} is not supported!')

        self.num_patches = (image_size // (4 * 2 * 2)) * (image_size //
                                                          (4 * 2 * 2))

    def forward(self, x):

        x = paddle.nn.functional.unfold(x,
                                        kernel_sizes=7,
                                        strides=4,
                                        paddings=2)

        x = x.transpose([0, 2, 1])

        x = self.attn1(x)
        B, HW, C = x.shape
        x = x.transpose([0, 2, 1])
        x = x.reshape([B, C, int(np.sqrt(HW)), int(np.sqrt(HW))])

        x = paddle.nn.functional.unfold(x,
                                        kernel_sizes=3,
                                        strides=2,
                                        paddings=1)
        x = x.transpose([0, 2, 1])

        x = self.attn2(x)
        B, HW, C = x.shape
        x = x.transpose([0, 2, 1])
        x = x.reshape([B, C, int(np.sqrt(HW)), int(np.sqrt(HW))])

        x = paddle.nn.functional.unfold(x,
                                        kernel_sizes=3,
                                        strides=2,
                                        paddings=1)
        x = x.transpose([0, 2, 1])

        x = self.proj(x)
        return x


class Mlp(nn.Layer):
    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 dropout=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)

        self.fc2 = nn.Linear(hidden_features, out_features)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Attention(nn.Layer):
    def __init__(self,
                 dim,
                 in_dim=None,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attention_dropout=0.,
                 dropout=0.,
                 skip_connection=False):
        super().__init__()
        self.num_heads = num_heads
        self.in_dim = in_dim or dim
        self.dim_head = dim // num_heads
        self.scale = qk_scale or self.dim_head**-0.5

        self.qkv = nn.Linear(dim, self.in_dim * 3)

        self.attn_dropout = nn.Dropout(attention_dropout)

        self.proj = nn.Linear(self.in_dim, self.in_dim)
        self.proj_dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(axis=-1)

        self.skip = skip_connection

    def transpose_multihead(self, x):
        if self.skip:  # token transformer
            new_shape = x.shape[:-1] + [self.num_heads, self.in_dim]
        else:  # regular attention
            new_shape = x.shape[:-1] + [self.num_heads, self.dim_head]
        x = x.reshape(new_shape)
        x = x.transpose([0, 2, 1, 3])
        return x

    def forward(self, x):
        B, H, C = x.shape
        qkv = self.qkv(x).chunk(3, axis=-1)
        q, k, v = map(self.transpose_multihead, qkv)
        q = q * self.scale
        attn = paddle.matmul(q, k, transpose_y=True)
        attn = self.softmax(attn)
        attn = self.attn_dropout(attn)

        z = paddle.matmul(attn, v)
        z = z.transpose([0, 2, 1, 3])
        if self.skip:  # token transformer
            z = z.reshape([B, -1, self.in_dim])
        else:  # regular attention
            z = z.reshape([B, -1, C])
        z = self.proj(z)
        z = self.proj_dropout(z)

        # skip connection
        if self.skip:
            z = z + v.squeeze(1)

        return z


class Block(nn.Layer):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 dropout=0.,
                 attention_dropout=0.,
                 droppath=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, epsilon=1e-6)
        self.attn = Attention(dim,
                              num_heads=num_heads,
                              qkv_bias=qkv_bias,
                              qk_scale=qk_scale,
                              dropout=dropout,
                              attention_dropout=attention_dropout)
        self.drop_path = DropPath(droppath) if droppath > 0. else Identity()
        self.norm2 = nn.LayerNorm(dim, epsilon=1e-6)
        self.mlp = Mlp(in_features=dim,
                       hidden_features=int(dim * mlp_ratio),
                       dropout=dropout)

    def forward(self, x):
        h = x
        x = self.norm1(x)
        x = self.attn(x)
        x = self.drop_path(x)
        x = h + x

        h = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = self.drop_path(x)
        x = h + x
        return x


class TokenPerformer(nn.Layer):
    def __init__(self, dim, in_dim, num_heads=1, kernel_ratio=0.5, dropout=0.1):
        super().__init__()
        self.embed_dim = in_dim * num_heads
        w_attr_1, b_attr_1 = self._init_weights()  # init for linear
        self.kqv = nn.Linear(dim,
                             3 * self.embed_dim,
                             weight_attr=w_attr_1,
                             bias_attr=b_attr_1)
        self.dropout = nn.Dropout(dropout)
        w_attr_2, b_attr_2 = self._init_weights()  # init for linear
        self.proj = nn.Linear(self.embed_dim,
                              self.embed_dim,
                              weight_attr=w_attr_2,
                              bias_attr=b_attr_2)
        self.num_heads = num_heads
        w_attr_3, b_attr_3 = self._init_weights_layernorm(
        )  # init for layernorm
        w_attr_4, b_attr_4 = self._init_weights_layernorm(
        )  # init for layernorm
        self.norm1 = nn.LayerNorm(dim,
                                  epsilon=1e-6,
                                  weight_attr=w_attr_3,
                                  bias_attr=b_attr_3)
        self.norm2 = nn.LayerNorm(self.embed_dim,
                                  epsilon=1e-6,
                                  weight_attr=w_attr_4,
                                  bias_attr=b_attr_4)

        w_attr_5, b_attr_5 = self._init_weights()  # init for linear
        w_attr_6, b_attr_6 = self._init_weights()  # init for linear
        self.mlp = nn.Sequential(
            nn.Linear(self.embed_dim,
                      self.embed_dim,
                      weight_attr=w_attr_5,
                      bias_attr=b_attr_5), nn.GELU(),
            nn.Linear(self.embed_dim,
                      self.embed_dim,
                      weight_attr=w_attr_6,
                      bias_attr=b_attr_6), nn.Dropout(dropout))

        self.m = int(self.embed_dim * kernel_ratio)

        self.w = np.random.random(size=(int(self.embed_dim * kernel_ratio),
                                        self.embed_dim))
        # init with orthognal matrix
        self.w = orthogonal(self.w)

        self.w = paddle.create_parameter(
            shape=[int(self.embed_dim * kernel_ratio), self.embed_dim],
            dtype='float32',
            default_initializer=nn.initializer.Assign(self.w /
                                                      math.sqrt(self.m)))

    def prm_exp(self, x):
        # x: [B, T, hs]
        # w: [m, hs]
        # return x: B, T, m
        xd = (x * x).sum(axis=-1, keepdim=True)
        xd = xd.expand([xd.shape[0], xd.shape[1], self.m]) / 2
        # same as einsum('bti,mi->btm', x, self.w)
        wtx = paddle.matmul(x, self.w, transpose_y=True)
        out = paddle.exp(wtx - xd) / math.sqrt(self.m)
        return out

    def single_attention(self, x):
        kqv = self.kqv(x).chunk(3, axis=-1)
        k, q, v = kqv[0], kqv[1], kqv[2]

        qp = self.prm_exp(q)
        kp = self.prm_exp(k)

        # same as einsum('bti,bi->bt, qp, kp.sum(axi=1).unsqueeze(2)')
        D = paddle.matmul(qp, kp.sum(axis=1).unsqueeze(2))
        # same as einsum('bti,bim->bnm')
        kptv = paddle.matmul(v, kp, transpose_x=True)
        # same as einsum('bti,bni->btn')
        y = paddle.matmul(qp, kptv, transpose_y=True)
        y = y / (D.expand([D.shape[0], D.shape[1], self.embed_dim]) + 1e-8)

        # skip connection
        y = self.proj(y)
        y = self.dropout(y)
        y = v + y
        return y

    def forward(self, x):
        x = self.norm1(x)
        x = self.single_attention(x)
        h = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = h + x
        return x


class TokenTransformer(nn.Layer):
    def __init__(self,
                 dim,
                 in_dim,
                 num_heads,
                 mlp_ratio=1.0,
                 qkv_bias=False,
                 qk_scale=None,
                 dropout=0.,
                 attention_dropout=0,
                 droppath=0.):
        super().__init__()
        w_attr_1, b_attr_1 = self._init_weights_layernorm()
        self.norm1 = nn.LayerNorm(dim, epsilon=1e-6)
        self.attn = Attention(dim,
                              in_dim=in_dim,
                              num_heads=num_heads,
                              qkv_bias=qkv_bias,
                              qk_scale=qk_scale,
                              dropout=dropout,
                              attention_dropout=attention_dropout,
                              skip_connection=True)
        self.drop_path = DropPath(droppath) if droppath > 0. else Identity()
        w_attr_2, b_attr_2 = self._init_weights_layernorm()
        self.norm2 = nn.LayerNorm(in_dim, epsilon=1e-6)
        self.mlp = Mlp(in_features=in_dim,
                       hidden_features=int(in_dim * mlp_ratio),
                       out_features=in_dim,
                       dropout=dropout)

    def forward(self, x):
        x = self.norm1(x)
        x = self.attn(x)
        h = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = self.drop_path(x)
        x = h + x
        return x


class T2TViT(nn.Layer):
    def __init__(self,
                 image_size=224,
                 in_channels=3,
                 num_classes=1000,
                 token_type='performer',
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 dropout=0.,
                 attention_dropout=0.,
                 droppath=0,
                 token_dim=64):
        super().__init__()
        self.num_classes = num_classes
        # convert image to paches: T2T-Module
        self.patch_embed = PatchEmbedding(image_size=image_size,
                                          token_type=token_type,
                                          in_channels=in_channels,
                                          embed_dim=embed_dim,
                                          token_dim=token_dim)
        num_patches = self.patch_embed.num_patches
        # tokens add for classification
        self.cls_token = paddle.create_parameter(
            shape=[1, 1, embed_dim],
            dtype='float32',
            default_initializer=trunc_normal_)
        # positional embeddings for patch positions
        self.pos_embed = paddle.create_parameter(
            shape=[1, num_patches + 1, embed_dim],
            dtype='float32',
            default_initializer=trunc_normal_)
        # dropout for positional embeddings
        self.pos_dropout = nn.Dropout(dropout)
        # droppath deacay rate
        depth_decay = paddle.linspace(0, droppath, depth)

        # craete self-attention layers
        layer_list = []
        for i in range(depth):
            block_layers = Block(dim=embed_dim,
                                 num_heads=num_heads,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias,
                                 qk_scale=qk_scale,
                                 dropout=dropout,
                                 attention_dropout=attention_dropout,
                                 droppath=depth_decay[i])
            layer_list.append(copy.deepcopy(block_layers))
        self.blocks = nn.LayerList(layer_list)

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
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand([x.shape[0], -1, -1])
        x = paddle.concat([cls_tokens, x], axis=1)
        x = x + self.pos_embed
        x = self.pos_dropout(x)

        # Self-Attention blocks
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        return x[:, 0]  # returns only cls_tokens

    def forward(self, x):
        x = self.forward_features(x)
        return x
