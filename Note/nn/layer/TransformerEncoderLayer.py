import tensorflow as tf
from Note.nn.layer.multihead_attention import multihead_attention
from Note.nn.layer.dense import dense
from Note.nn.layer.layer_normalization import layer_normalization
from Note.nn.layer.dropout import dropout
from Note.nn.activation import activation_dict


class TransformerEncoderLayer:
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout_rate: float = 0.1,
                 activation = tf.nn.relu,
                 layer_norm_eps: float = 1e-5, norm_first: bool = False,
                 bias: bool = True, dtype='float32'):
        self.self_attn = multihead_attention(nhead, input_size=d_model, dtype=dtype)
        # Implementation of Feedforward model
        self.linear1 = dense(dim_feedforward, d_model, use_bias=bias, dtype=dtype)
        self.dropout = dropout(dropout_rate)
        self.linear2 = dense(d_model, dim_feedforward, use_bias=bias, dtype=dtype)

        self.norm_first = norm_first
        self.norm1 = layer_normalization(d_model, epsilon=layer_norm_eps, dtype=dtype)
        self.norm2 = layer_normalization(d_model, epsilon=layer_norm_eps, dtype=dtype)
        self.dropout1 = dropout(dropout_rate)
        self.dropout2 = dropout(dropout_rate)

        if isinstance(activation, str):
            activation = activation_dict[activation]
        else:
            self.activation = activation


    def output(
            self,
            src,
            src_mask=None,
            ):

        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1.output(x), src_mask)
            x = x + self._ff_block(self.norm2.output(x))
        else:
            x = self.norm1.output(x + self._sa_block(x, src_mask))
            x = self.norm2.output(x + self._ff_block(x))

        return x


    # self-attention block
    def _sa_block(self, x,
                  attn_mask=None):
        x = self.self_attn.output(x,
                           mask=attn_mask,
                           )[0]
        return self.dropout1.output(x)


    # feed forward block
    def _ff_block(self, x):
        x = self.linear2.output(self.dropout.output(self.activation(self.linear1.output(x))))
        return self.dropout2.output(x)