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

from typing import List, Tuple, Union
import math
import numpy as np

import paddle
import paddle.nn as nn
from ..vision_transformer import Attention, Block, VisionTransformer


def do_nothing(x, mode=None):
    return x


def bipartite_soft_matching(
        metric,
        r,
        class_token=False,
        distill_token=False, ):
    """
    Applies ToMe with a balanced matching set (50%, 50%).

    Input size is [batch, tokens, channels].
    r indicates the number of tokens to remove (max 50% of tokens).

    Extra args:
     - class_token: Whether or not there's a class token.
     - distill_token: Whether or not there's also a distillation token.

    When enabled, the class token and distillation tokens won't get merged.
    """
    protected = 0
    if class_token:
        protected += 1
    if distill_token:
        protected += 1

    # We can only reduce by a maximum of 50% tokens
    t = metric.shape[1]
    r = min(r, (t - protected) // 2)

    if r <= 0:
        return do_nothing, do_nothing

    with paddle.no_grad():
        metric = metric / metric.norm(axis=-1, keepdim=True)
        a, b = metric[..., ::2, :], metric[..., 1::2, :]
        scores = a @b.transpose((0, 2, 1))

        if class_token:
            scores[..., 0, :] = -math.inf
        if distill_token:
            scores[..., :, 0] = -math.inf

        node_max = scores.max(axis=-1)
        node_idx = scores.argmax(axis=-1)
        edge_idx = node_max.argsort(axis=-1, descending=True)[..., None]

        unm_idx = edge_idx[..., r:, :]  # Unmerged Tokens
        src_idx = edge_idx[..., :r, :]  # Merged Tokens
        dst_idx = node_idx[..., None].take_along_axis(axis=-2, indices=src_idx)

        if class_token:
            # Sort to ensure the class token is at the start
            unm_idx = unm_idx.sort(axis=1)

    def merge(x, mode="mean"):
        src, dst = x[..., ::2, :], x[..., 1::2, :]
        n, t1, c = src.shape
        unm = src.take_along_axis(
            axis=-2, indices=unm_idx.expand((n, t1 - r, c)))
        src = src.take_along_axis(axis=-2, indices=src_idx.expand((n, r, c)))

        # dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)
        assert mode == 'sum', "only support mode == 'sum'"
        dst.put_along_axis_(
            axis=-2,
            indices=dst_idx.expand((n, r, c)),
            values=src,
            reduce='add')

        if distill_token:
            return paddle.concat(
                [unm[:, :1], dst[:, :1], unm[:, 1:], dst[:, 1:]], axis=1)
        else:
            return paddle.concat([unm, dst], axis=1)

    def unmerge(x):
        unm_len = unm_idx.shape[1]
        unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]
        n, _, c = unm.shape

        src = dst.take_along_axis(axis=-2, indices=dst_idx.expand((n, r, c)))

        out = paddle.zeros((n, metric.shape[1], c), dtype=x.dtype)

        out[..., 1::2, :] = dst
        out.put_along_axis_(
            axis=-2, indices=(2 * unm_idx).expand((n, unm_len, c)), values=unm)
        out.put_along_axis_(
            axis=-2, indices=(2 * src_idx).expand((n, r, c)), values=src)

        return out

    return merge, unmerge


def merge_wavg(merge, x, size=None):
    """
    Applies the merge function by taking a weighted average based on token size.
    Returns the merged tensor and the new token sizes.
    """
    if size is None:
        size = paddle.ones_like(x[..., 0, None])

    x = merge(x * size, mode="sum")
    size = merge(size, mode="sum")

    x = x / size
    return x, size


class ToMeBlock(Block):
    """
    Modifications:
     - Apply ToMe between the attention and mlp blocks
     - Compute and propogate token size and potentially the token sources.
    """

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        # Note: this is copied from passl.models.vision_transformer.Block with modifications.
        attn_size = self._tome_info["size"] if self._tome_info[
            "prop_attn"] else None
        x_attn, metric = self.attn(self.norm1(x), attn_size)
        x = x + self.drop_path(x_attn)

        r = self._tome_info["r"].pop(0)
        if r > 0:
            # Apply ToMe here
            merge, _ = bipartite_soft_matching(
                metric,
                r,
                self._tome_info["class_token"],
                self._tome_info["distill_token"], )
            x, self._tome_info["size"] = merge_wavg(merge, x,
                                                    self._tome_info["size"])

        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class ToMeAttention(Attention):
    """
    Modifications:
     - Apply proportional attention
     - Return the mean of k over heads from attention
    """

    def forward(
            self, x: paddle.Tensor,
            size: paddle.Tensor=None) -> Tuple[paddle.Tensor, paddle.Tensor]:
        # Note: this is copied from passl.models.vision_transformer.Attention with modifications.
        N, C = x.shape[1:]
        qkv = self.qkv(x).reshape((-1, N, 3, self.num_heads, C //
                                   self.num_heads)).transpose((2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q.matmul(k.transpose((0, 1, 3, 2)))) * self.scale

        # Apply proportional attention
        if size is not None:
            attn = attn + size.log()[:, None, None, :, 0]

        attn = nn.functional.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)

        x = (attn.matmul(v)).transpose((0, 2, 1, 3)).reshape((-1, N, C))
        x = self.proj(x)
        x = self.proj_drop(x)

        # Return k as well here
        return x, k.mean(1)


def parse_r(num_layers: int,
            r: Union[List[int], Tuple[int, float], int]) -> List[int]:
    """
    Process a constant r or r schedule into a list for use internally.

    r can take the following forms:
     - int: A constant number of tokens per layer.
     - Tuple[int, float]: A pair of r, inflection.
       Inflection describes there the the reduction / layer should trend
       upward (+1), downward (-1), or stay constant (0). A value of (r, 0)
       is as providing a constant r. (r, -1) is what we describe in the paper
       as "decreasing schedule". Any value between -1 and +1 is accepted.
     - List[int]: A specific number of tokens per layer. For extreme granularity.
    """
    inflect = 0
    if isinstance(r, list):
        if len(r) < num_layers:
            r = r + [0] * (num_layers - len(r))
        return list(r)
    elif isinstance(r, tuple):
        r, inflect = r

    min_val = int(r * (1.0 - inflect))
    max_val = 2 * r - min_val
    step = (max_val - min_val) / (num_layers - 1)

    return [int(min_val + step * i) for i in range(num_layers)]


def make_tome_class(transformer_class):
    class ToMeVisionTransformer(transformer_class):
        """
        Modifications:
        - Initialize r, token size, and token sources.
        """

        def forward(self, *args, **kwdargs):
            self._tome_info["r"] = parse_r(len(self.blocks), self.r)
            self._tome_info["size"] = None

            return super().forward(*args, **kwdargs)

        def forward_features(self, x):
            B = x.shape[0]
            x = self.patch_embed(x)

            T = x.shape[1]

            cls_tokens = self.cls_token.expand(
                (B, -1, -1))  # stole cls_tokens impl from Phil Wang, thanks
            x = paddle.concat((cls_tokens, x), axis=1)
            x = x + self.pos_embed
            x = self.pos_drop(x)

            for blk in self.blocks:
                x = blk(x)

            if self.global_pool:
                # ---- ToMe changes this ----
                # Global average pool proportional to token size
                if self._tome_info["size"] is not None:
                    x = (x * self._tome_info["size"])[:, 1:, :].sum(axis=1) / T
                else:
                    x = x[:, 1:, :].mean(
                        axis=1)  # global pool without cls token
                # ---- End of change ----

                outcome = self.fc_norm(x)
            else:
                x = self.norm(x)
                outcome = x[:, 0]

            return outcome

    return ToMeVisionTransformer


def apply_patch(model: VisionTransformer, prop_attn: bool=False):
    """
    Applies ToMe to this transformer. Afterward, set r using model.r.

    For proportional attention, set prop_attn to True. This is only necessary when evaluating models off
    the shelf. For trianing and for evaluating MAE models off the self set this to be False.
    """
    ToMeVisionTransformer = make_tome_class(model.__class__)

    model.__class__ = ToMeVisionTransformer
    model.r = 0
    model._tome_info = {
        "r": model.r,
        "size": None,
        "prop_attn": prop_attn,
        "class_token": model.cls_token is not None,
        "distill_token": False,
    }

    if hasattr(model, "dist_token") and model.dist_token is not None:
        model._tome_info["distill_token"] = True

    for layer in model.sublayers():
        if isinstance(layer, Block):
            layer.__class__ = ToMeBlock
            layer._tome_info = model._tome_info
        elif isinstance(layer, Attention):
            layer.__class__ = ToMeAttention
