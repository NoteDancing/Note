from dataclasses import dataclass
from typing import Dict, Optional, Union

import tensorflow as tf
from Note import nn



@dataclass
class ModelArgs:
    model_type: str
    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int
    num_attention_heads: int
    rms_norm_eps: float
    vocab_size: int
    num_key_value_heads: int = None
    rope_theta: float = 10000
    rope_traditional: bool = False
    rope_scaling: Optional[Dict[str, Union[float, str]]] = None

    def __post_init__(self):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads

        if self.rope_scaling:
            required_keys = {"long_factor", "type"}
            if not all(key in self.rope_scaling for key in required_keys):
                raise ValueError(f"rope_scaling must contain keys {required_keys}")

            if self.rope_scaling["type"] != "linear":
                print(
                    "[WARNING] rope_scaling 'type' currently only supports 'linear' setting rope scaling to false."
                )
                self.rope_scaling = None


class Attention:
    def __init__(self, args: ModelArgs):
        dim = args.hidden_size
        self.n_heads = n_heads = args.num_attention_heads
        self.n_kv_heads = n_kv_heads = args.num_key_value_heads

        head_dim = args.hidden_size // n_heads
        self.scale = head_dim**-0.5

        op_size = n_heads * head_dim + 2 * (n_kv_heads * head_dim)
        self.qkv_proj = nn.dense(op_size, dim, use_bias=False)
        self.o_proj = nn.dense(dim, n_heads * head_dim, use_bias=False)

        rope_scale = (
            1 / args.rope_scaling["factor"]
            if args.rope_scaling is not None and args.rope_scaling["type"] == "linear"
            else 1
        )
        self.rope = nn.RoPE(
            head_dim,
            traditional=args.rope_traditional,
            base=args.rope_theta,
            scale=rope_scale,
        )

    def __call__(
        self,
        x,
        mask = None,
        cache = None,
    ):
        B, L, D = x.shape

        qkv = self.qkv_proj(x)
        queries, keys, values = tf.split(qkv, 3, axis=-1)

        # Prepare the queries, keys and values for the attention computation
        queries = tf.transpose(tf.reshape(queries, (B, L, self.n_heads, -1)), (0, 2, 1, 3))
        keys = tf.transpose(tf.reshape(keys, (B, L, self.n_kv_heads, -1)), (0, 2, 1, 3))
        values = tf.transpose(tf.reshape(values, (B, L, self.n_kv_heads, -1)), (0, 2, 1, 3))

        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        output = nn.scaled_dot_product_attention(
            queries, keys, values, scale=self.scale, mask=mask
        )
        output = tf.reshape(tf.transpose(output, (0, 2, 1, 3)), (B, L, -1))
        return self.o_proj(output)


class MLP:
    def __init__(self, dim, hidden_dim):
        self.gate_up_proj = nn.dense(2 * hidden_dim, dim, use_bias=False)
        self.down_proj = nn.dense(dim, hidden_dim, use_bias=False)

    def __call__(self, x):
        x = self.gate_up_proj(x)
        gate, x = tf.split(x, 2, axis=-1)
        return self.down_proj(tf.nn.silu(gate) * x)


class TransformerBlock:
    def __init__(self, args: ModelArgs):
        self.num_attention_heads = args.num_attention_heads
        self.hidden_size = args.hidden_size
        self.self_attn = Attention(args)
        self.mlp = MLP(args.hidden_size, args.intermediate_size)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )
        self.args = args

    def __call__(
        self,
        x,
        mask = None,
        cache = None,
    ):
        r = self.self_attn(self.input_layernorm(x), mask, cache)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        out = h + r
        return out


class Phi3Model:
    def __init__(self, args: ModelArgs):
        self.args = args
        self.vocab_size = args.vocab_size
        self.num_hidden_layers = args.num_hidden_layers
        assert self.vocab_size > 0
        self.embed_tokens = nn.embedding(args.hidden_size, args.vocab_size)
        self.layers = [
            TransformerBlock(args=args) for _ in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        inputs,
        cache=None,
    ):
        h = self.embed_tokens(inputs)

        mask = None
        if h.shape[1] > 1:
            mask = nn.create_additive_causal_mask(h.shape[1])
            mask = tf.cast(mask, h.dtype)

        if cache is None:
            cache = [None] * len(self.layers)

        for layer, c in zip(self.layers, cache):
            h = layer(h, mask, c)

        return self.norm(h)


class Phi3(nn.Model):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.model_type = args.model_type
        self.model = Phi3Model(args)
        self.head = self.dense(args.vocab_size, args.hidden_size, use_bias=False)
        self.args = args

    def __call__(
        self,
        inputs,
        cache=None,
    ):
        out = self.model(inputs, cache)
        return self.head(out)

    @property
    def layers(self):
        return self.model.layers

    @property
    def head_dim(self):
        return self.args.hidden_size // self.args.num_attention_heads

    @property
    def n_kv_heads(self):
        return self.args.num_key_value_heads