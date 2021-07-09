import paddle
from paddle import tensor
from paddle.fluid import layers
import paddle.nn.functional as F
from paddle.nn import Layer, Linear
from paddle.fluid.data_feeder import convert_dtype

class AttentionPool2D(Layer):
    def __init__(self, spacial_dim, embed_dim, num_heads, output_dim,
                 dropout=0,
                 need_weights=False,
                 weight_attr=None,
                 bias_attr=None):
        super(AttentionPool2D, self).__init__()
        self.positional_embedding = paddle.randn((spacial_dim ** 2 + 1, embed_dim)) / embed_dim ** 0.5
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.need_weights = need_weights

        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.in_features = embed_dim
        self.q_proj = Linear(
            embed_dim, embed_dim, weight_attr, bias_attr=bias_attr)
        self.k_proj = Linear(
            embed_dim, embed_dim, weight_attr, bias_attr=bias_attr)
        self.v_proj = Linear(
            embed_dim, embed_dim, weight_attr, bias_attr=bias_attr)
        self.out_proj = Linear(
            embed_dim, output_dim or embed_dim, weight_attr, bias_attr=bias_attr)

    def _prepare_qkv(self, query, key, value):
        q = self.q_proj(query)
        q = tensor.reshape(x=q, shape=[0, 0, self.num_heads, self.head_dim])
        q = tensor.transpose(x=q, perm=[0, 2, 1, 3])
        k, v = self.compute_kv(key, value)
        return (q, k, v)
    
    def compute_kv(self, key, value):
        k = self.k_proj(key)
        v = self.v_proj(value)
        k = tensor.reshape(x=k, shape=[0, 0, self.num_heads, self.head_dim])
        k = tensor.transpose(x=k, perm=[0, 2, 1, 3])
        v = tensor.reshape(x=v, shape=[0, 0, self.num_heads, self.head_dim])
        v = tensor.transpose(x=v, perm=[0, 2, 1, 3])
        return k, v
    

    def forward(self, x):
        x = x.reshape((x.shape[0], x.shape[1], x.shape[2] * x.shape[3])).transpose((2, 0, 1))  # NCHW -> (HW)NC
        x = paddle.concat([x.mean(axis=0, keepdim=True), x], axis=0)  # (HW+1)NC
        x = x + self.positional_embedding.unsqueeze(axis=1).astype(x.dtype)  # (HW+1)NC
        x = self.multi_head_attention_forward(
            query=x, key=x, value=x,
        )
        return x[0]



    def multi_head_attention_forward(self,
        query, key=None, value=None,
        attn_mask=None, cache=None):
        key = query if key is None else key
        value = query if value is None else value
        # compute q ,k ,v
        q, k, v = self._prepare_qkv(query, key, value)
    
        # scale dot product attention
        # TODO(guosheng): use tensor.matmul, however it doesn't support `alpha`
        product = layers.matmul(
            x=q, y=k, transpose_y=True, alpha=self.head_dim**-0.5)
        if attn_mask is not None:
            # Support bool or int mask
            attn_mask = _convert_attention_mask(attn_mask, product.dtype)
            product = product + attn_mask
        weights = F.softmax(product)
        if self.dropout:
            weights = F.dropout(
                weights,
                self.dropout,
                training=self.training,
                mode="upscale_in_train")
    
        out = tensor.matmul(weights, v)
    
        # combine heads
        out = tensor.transpose(out, perm=[0, 2, 1, 3])
        out = tensor.reshape(x=out, shape=[0, 0, out.shape[2] * out.shape[3]])
    
        # project to output
        out = self.out_proj(out)
    
        outs = [out]
        if self.need_weights:
            outs.append(weights)
        if cache is not None:
            outs.append(cache)
        return out if len(outs) == 1 else tuple(outs)

def _convert_attention_mask(attn_mask, dtype):
    if attn_mask is not None and attn_mask.dtype != dtype:
        attn_mask_dtype = convert_dtype(attn_mask.dtype)
        if attn_mask_dtype == 'bool' or 'int' in attn_mask_dtype:
            attn_mask = (paddle.cast(attn_mask, dtype) - 1.0) * 1e9
        else:
            attn_mask = paddle.cast(attn_mask, dtype)
    return attn_mask