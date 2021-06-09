import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn.layer.common import Linear, Dropout
from paddle.nn.layer.norm import LayerNorm
from paddle.fluid.dygraph import Layer, LayerList
from paddle.nn.layer.transformer import MultiHeadAttention 
from paddle.nn.layer.transformer import _convert_attention_mask
from paddle.nn.layer.transformer import _convert_param_attr_to_list


class QuickGELU(Layer):
    def forward(self, x):
        return x * F.sigmoid(1.702 * x)


class ResidualAttentionBlock(Layer):
    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward,
                 dropout=0.1,
                 activation="relu",
                 attn_dropout=None,
                 act_dropout=None,
                 normalize_before=False,
                 weight_attr=None,
                 bias_attr=None):
        self._config = locals()
        self._config.pop("self")
        self._config.pop("__class__", None)  # py3

        super(ResidualAttentionBlock, self).__init__()
        attn_dropout = dropout if attn_dropout is None else attn_dropout
        act_dropout = dropout if act_dropout is None else act_dropout
        self.normalize_before = normalize_before

        weight_attrs = _convert_param_attr_to_list(weight_attr, 2)
        bias_attrs = _convert_param_attr_to_list(bias_attr, 2)

        self.attn = MultiHeadAttention(
            d_model,
            nhead,
            dropout=attn_dropout,
            weight_attr=weight_attrs[0],
            bias_attr=bias_attrs[0])
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(
            ("c_fc", Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", Linear(d_model * 4, d_model))
        )
        self.ln_2 = LayerNorm(d_model)

    def forward(self, src, src_mask=None, cache=None):
        src_mask = _convert_attention_mask(src_mask, src.dtype)

        residual = src
        src = self.ln_1(src)
        # Add cache for encoder for the usage like UniLM
        if cache is None:
            src = self.attn(src, src, src, src_mask)
        else:
            src, incremental_cache = self.attn(src, src, src, src_mask,
                                                    cache)

        src = residual + src

        residual = src
        src = self.mlp(self.ln_2(src))
        src = residual + src
        return src if cache is None else (src, incremental_cache)

    def gen_cache(self, src):
        incremental_cache = self.attn.gen_cache(
            src, type=self.attn.Cache)
        return incremental_cache


class TransformerEncoder(Layer):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = LayerList([(encoder_layer if i == 0 else
                                  type(encoder_layer)(**encoder_layer._config))
                                 for i in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, src_mask=None, cache=None):
        src_mask = _convert_attention_mask(src_mask, src.dtype)

        output = src
        new_caches = []
        for i, mod in enumerate(self.layers):
            if cache is None:
                output = mod(output, src_mask=src_mask)
            else:
                output, new_cache = mod(output,
                                        src_mask=src_mask,
                                        cache=cache[i])
                new_caches.append(new_cache)

        if self.norm is not None:
            output = self.norm(output)

        return output if cache is None else (output, new_caches)

    def gen_cache(self, src):
        cache = [layer.gen_cache(src) for layer in self.layers]
        return cache



class Transformer(Layer):
    def __init__(self,
                 d_model=512,
                 nhead=8,
                 layers=6,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation="relu",
                 attn_dropout=None,
                 act_dropout=None,
                 normalize_before=False,
                 weight_attr=None,
                 bias_attr=None,
                 custom_encoder=None,
                 custom_decoder=None):
        super(Transformer, self).__init__()

        self.width = d_model
        self.layers = layers
        if isinstance(bias_attr, (list, tuple)):
            if len(bias_attr) == 1:
                encoder_bias_attr = [bias_attr[0]] * 2
            elif len(bias_attr) == 2:
                encoder_bias_attr = bias_attr
            elif len(bias_attr) == 3:
                encoder_bias_attr = [bias_attr[0], bias_attr[-1]]
            else:
                assert False, (
                    "length of bias_attr should be 1 or 2 or 3 when it is a list/tuple"
                )
        else:
            encoder_bias_attr = bias_attr

        if isinstance(weight_attr, (list, tuple)):
            if len(weight_attr) == 1:
                encoder_weight_attr = [weight_attr[0]] * 2
            elif len(weight_attr) == 2:
                encoder_weight_attr = weight_attr
            elif len(weight_attr) == 3:
                encoder_weight_attr = [weight_attr[0], weight_attr[-1]]
            else:
                assert False, (
                    "length of weight_attr should be 1 or 2 or 3 when it is a list/tuple"
                )
        else:
            encoder_weight_attr = weight_attr

        if custom_encoder is not None:
            self.encoder = custom_encoder
        else:
            encoder_layer = ResidualAttentionBlock(
                d_model, nhead, dim_feedforward, dropout, activation,
                attn_dropout, act_dropout, normalize_before,
                encoder_weight_attr, encoder_bias_attr)
            encoder_norm = LayerNorm(d_model)
            self.encoder = TransformerEncoder(encoder_layer, layers,
                                              encoder_norm)


        self.d_model = d_model
        self.nhead = nhead

    def forward(self, src, src_mask=None, tgt_mask=None, memory_mask=None):
        src_mask = _convert_attention_mask(src_mask, src.dtype)
        return self.encoder(src, src_mask=src_mask)

    def generate_square_subsequent_mask(self, length):
        return paddle.tensor.triu(
            (paddle.ones(
                (length, length), dtype=paddle.get_default_dtype()) * -np.inf),
            1)
