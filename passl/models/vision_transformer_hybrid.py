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

from collections.abc import Callable

import scipy
import os
import numpy as np
import paddle
import paddle.nn as nn
import paddle.distributed as dist

from passl.utils import logger
from passl.models.base_model import Model
from passl.nn import init
from passl.models.vision_transformer import DropPath, PatchEmbed

from passl.distributed import distributed_env as dist_env
from passl.nn import FinerGrainedRowParallelLinear, FinerGrainedColumnParallelLinear

__all__ = [
    'ViT_hybrid_base_patch16_224',
    'VisionTransformerHybrid',
]


class MlpHybrid(nn.Layer):
    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.,
                 input_is_parallel=False,
                 gather_output=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        # self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc1 = FinerGrainedColumnParallelLinear(in_features, hidden_features, input_is_parallel=True, gather_output=False)
        self.act = act_layer()
        # self.fc2 = nn.Linear(hidden_features, out_features)
        self.fc2 = FinerGrainedRowParallelLinear(hidden_features, out_features, input_is_parallel=True, gather_output=False)
        self.drop = nn.Dropout(drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            init.xavier_uniform_(m.weight)
            if m.bias is not None:
                init.normal_(m.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class AttentionHybrid(nn.Layer):
    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 input_is_parallel=False,
                 gather_output=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        # self.qkv = nn.Linear(dim, dim * 3, bias_attr=qkv_bias)
        # Note(GuoxiaWang): we can use columnsharded_linear or rowsharded_linear according communication volume.
        # self.qkv = FinerGrainedColumnParallelLinear(dim, dim * 3, bias_attr=qkv_bias, input_is_parallel=True, gather_output=False)
        self.qkv = FinerGrainedRowParallelLinear(dim, dim * 3, bias_attr=qkv_bias, input_is_parallel=True, gather_output=False)
        self.attn_drop = nn.Dropout(attn_drop)
        # self.proj = nn.Linear(dim, dim)
        self.proj = FinerGrainedRowParallelLinear(dim, dim, input_is_parallel=True, gather_output=False)
        self.proj_drop = nn.Dropout(proj_drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            init.xavier_uniform_(m.weight)
            if m.bias is not None:
                init.zeros_(m.bias)

    def forward(self, x):
        # B= paddle.shape(x)[0]
        N, C = x.shape[1:]
        qkv = self.qkv(x).reshape((-1, N, 3, self.num_heads, C //
                                   self.num_heads)).transpose((2, 0, 3, 1, 4))

        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q.matmul(k.transpose((0, 1, 3, 2)))) * self.scale
        attn = nn.functional.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)

        x = (attn.matmul(v)).transpose((0, 2, 1, 3)).reshape((-1, N, C))
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class BlockHybird(nn.Layer):
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
                 norm_layer='nn.LayerNorm',
                 epsilon=1e-5,
                 input_is_parallel=False,
                 gather_output=False):
        super().__init__()

        if isinstance(norm_layer, str):
            self.norm1 = eval(norm_layer)(dim, epsilon=epsilon)
        elif isinstance(norm_layer, Callable):
            self.norm1 = norm_layer(dim)
        else:
            raise TypeError(
                "The norm_layer must be str or paddle.nn.layer.Layer class")
        self.attn = AttentionHybrid(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            input_is_parallel=input_is_parallel,
            gather_output=gather_output)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        if isinstance(norm_layer, str):
            self.norm2 = eval(norm_layer)(dim, epsilon=epsilon)
        elif isinstance(norm_layer, Callable):
            self.norm2 = norm_layer(dim)
        else:
            raise TypeError(
                "The norm_layer must be str or paddle.nn.layer.Layer class")
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MlpHybrid(in_features=dim,
                       hidden_features=mlp_hidden_dim,
                       act_layer=act_layer,
                       drop=drop,
                       input_is_parallel=input_is_parallel,
                       gather_output=gather_output)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x



class VisionTransformerHybrid(Model):
    """ Vision Transformer with support for patch input
    """

    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_chans=3,
                 class_num=1000,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_layer='nn.LayerNorm',
                 epsilon=1e-5,
                 representation_size=None,
                 **kwargs):
        super().__init__()
        self.class_num = class_num
        self.representation_size = representation_size

        self.num_features = self.embed_dim = embed_dim

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.pos_embed = self.create_parameter(
            shape=(1, num_patches + 1, embed_dim),
            default_initializer=paddle.nn.initializer.Constant(value=0.))
        self.cls_token = self.create_parameter(
            shape=(1, 1, embed_dim),
            default_initializer=paddle.nn.initializer.Constant(value=0.))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = np.linspace(0, drop_path_rate, depth)

        self.blocks = nn.LayerList([
            BlockHybird(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                epsilon=epsilon,
                input_is_parallel=False if i==0 else True,
                gather_output=True if i==depth-1 else False) for i in range(depth)
        ])

        if isinstance(norm_layer, str):
            self.norm = eval(norm_layer)(embed_dim, epsilon=epsilon)
        elif isinstance(norm_layer, Callable):
            self.norm = norm_layer(embed_dim)
        else:
            raise TypeError(
                "The norm_layer must be str or paddle.nn.layer.Layer class")

        # Classifier head
        if self.representation_size is not None:
            self.head0 = nn.Linear(embed_dim, representation_size)
            self.tanh = nn.Tanh()
            self.head = nn.Linear(
                representation_size,
                class_num) if class_num > 0 else nn.Identity()
            init.xavier_uniform_(self.head0.weight)
            init.zeros_(self.head0.bias)
            init.xavier_uniform_(self.head.weight)
            init.constant_(self.head.bias, -10.0)
        else:
            self.head = nn.Linear(
                embed_dim, class_num) if class_num > 0 else nn.Identity()
            init.zeros_(self.head.weight)
            init.zeros_(self.head.bias)

        init.normal_(self.pos_embed, std=0.02)
        init.zeros_(self.cls_token)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.LayerNorm):
            init.zeros_(m.bias)
            init.ones_(m.weight)

    def forward_features(self, x):
        # B = x.shape[0]
        B = paddle.shape(x)[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand((B, -1, -1))
        x = paddle.concat((cls_tokens, x), axis=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        mp_rank = dist_env.get_model_parallel_world_rank()
        mp_world_size = dist_env.get_model_parallel_world_size()
        mp_group = dist_env.get_model_parallel_group()

        x = paddle.split(x, mp_world_size, axis=0)[mp_rank]
        for blk in self.blocks:
            x = blk(x)

        tensor_list = []
        dist.all_gather(tensor_list, x, group=mp_group)
        x = paddle.concat(tensor_list, axis=0)

        x = self.norm(x)
        return x[:, 0]

    def forward(self, x):
        x = self.forward_features(x)
        if self.representation_size is not None:
            x = self.tanh(self.head0(x))
        x = self.head(x)
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

        mp_rank = dist_env.get_model_parallel_world_rank()
        mp_world_size = dist_env.get_model_parallel_world_size()

        for key, value in param_state_dict.items():
            if key in param_state_dict and key in state_dict and param_state_dict[
                    key].shape != state_dict[key].shape:
                if param_state_dict[key].shape[0] != state_dict[key].shape[0]:
                    param_state_dict[key] = paddle.split(param_state_dict[key], mp_world_size, axis=0)[mp_rank]
                elif param_state_dict[key].shape[1] != state_dict[key].shape[1]:
                    param_state_dict[key] = paddle.split(param_state_dict[key], mp_world_size, axis=1)[mp_rank]

        if not finetune:
            self.set_dict(param_state_dict)
            return

        for k in ['head0.weight', 'head0.bias', 'head.weight', 'head.bias']:
            if k in param_state_dict:
                logger.info(f"Removing key {k} from pretrained checkpoint")
                del param_state_dict[k]

        # interpolate position embedding
        pos_embed_ckt = param_state_dict['pos_embed']  # [1, N, 1024]
        embedding_size = pos_embed_ckt.shape[-1]
        num_patches = self.patch_embed.num_patches
        num_extra_tokens = self.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_ckt.shape[-2] - num_extra_tokens)**0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches**0.5)
        # class_token and dist_token are kept unchanged
        extra_tokens = pos_embed_ckt[:, :num_extra_tokens]
        # only the position tokens are interpolated
        pos_tokens = pos_embed_ckt[0, num_extra_tokens:]
        # pos_tokens = pos_tokens.reshape([orig_size, orig_size, embedding_size])

        # zoom = (new_size / orig_size, new_size / orig_size, 1)
        # pos_tokens_new = scipy.ndimage.zoom(pos_tokens.numpy(), zoom, order=1)
        # pos_tokens = paddle.to_tensor(pos_tokens_new, dtype=pos_tokens.dtype)
        # pos_tokens = pos_tokens.reshape([1, new_size*new_size, embedding_size])
        # new_pos_embed = paddle.concat((extra_tokens, pos_tokens), axis=1)
        # param_state_dict['pos_embed'] = new_pos_embed

        pos_tokens = paddle.transpose(
            pos_tokens.reshape([-1, orig_size, orig_size, embedding_size]),
            perm=[0, 3, 1, 2])
        dtype = pos_tokens.dtype
        pos_tokens = paddle.nn.functional.interpolate(
            pos_tokens.astype(paddle.float32),
            size=(new_size, new_size),
            mode='bicubic',
            align_corners=False).astype(dtype)
        pos_tokens = paddle.transpose(
            pos_tokens, perm=[0, 2, 3, 1]).flatten(1, 2)
        new_pos_embed = paddle.concat((extra_tokens, pos_tokens), axis=1)
        param_state_dict['pos_embed'] = new_pos_embed

        self.set_dict(param_state_dict)
        return

    def save(self, path, local_rank=0, rank=0):
        paddle.save(self.state_dict(), path + ".pdparams")


def ViT_hybrid_base_patch16_224(**kwargs):
    model = VisionTransformerHybrid(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        epsilon=1e-6,
        representation_size=768,
        **kwargs)
    return model
