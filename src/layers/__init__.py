import tensorflow as tf

from typing import *
from .utils import update_gradients, scaled_dot_product_attention, positional_encoding
from src.initializers import SingularValueInitializer

"""
Decomposition based layers
"""


class SVDDense(tf.keras.layers.Layer):
    """SVD based densely connected layer."""

    def __init__(self, units: int, rank: int = None, activation: str = 'relu', use_bias: bool = True):
        super(SVDDense, self).__init__()
        # initialise parameters
        self.units = units
        self.rank = tf.cast(units / 2, tf.int32) if rank is None else rank
        self.activation = tf.keras.activations.get(activation)
        self.use_bias = use_bias

        # initialise variables
        self._u = None
        self._s = None
        self._v = None
        self._bias = None

    def build(self, input_shape: tf.TensorShape):
        # define shapes
        u_shape = tf.TensorShape([input_shape[-1], self.rank])
        s_shape = tf.TensorShape([self.rank])
        v_shape = tf.TensorShape([self.units, self.rank])
        bias_shape = tf.TensorShape([self.units])
        # define initializers
        o_initializer = tf.keras.initializers.Orthogonal
        s_initializer = SingularValueInitializer(input_shape[-1], self.units)
        z_initializer = tf.keras.initializers.get('Zeros')
        # define variables
        self._u = self.add_weight("U", shape=u_shape, dtype=tf.float32, initializer=o_initializer)
        self._s = self.add_weight("S", shape=s_shape, dtype=tf.float32, initializer=s_initializer)
        self._v = self.add_weight("V", shape=v_shape, dtype=tf.float32, initializer=o_initializer)
        if self.use_bias:
            self._bias = self.add_weight("bias", shape=bias_shape, dtype=tf.float32, initializer=z_initializer)

    def __call__(self, inputs: tf.Tensor):
        kernel = self._u @ tf.linalg.diag(self._s) @ tf.transpose(self._v)
        outputs = inputs @ kernel
        if self.use_bias:
            outputs += self._bias
        # activate
        return self.activation(outputs)


class CASVDDenseMul(tf.keras.layers.Layer):
    """src based densely connected layer."""

    def __init__(self,
                 units: int,
                 rank: int,
                 activation: str,
                 use_bias: bool = True,
                 context_activation: callable = tf.nn.sigmoid
                 ):
        """Initialise layer.

        Parameters
        ----------
        units: int
            number of nodes
        rank: int
            rank of decomposition (<= units)
        activation: str
            Activation function name
        use_bias: bool
            Whether to add bias
        context_activation
            Context activation function
        """
        super(CASVDDenseMul, self).__init__(self)
        # initialise parameters
        self.units = units
        self.rank = rank
        self.activation = tf.keras.activations.get(activation)
        self.use_bias = use_bias
        self.context_activation = context_activation

        # initialise variables
        self.u = None
        self.s = None
        self.v = None
        self.bias = None
        self.context_bias = None
        self.context_kernel = None

    def __build__(self, input_shapes: List[tf.TensorShape]):
        # unpack input shapes
        assert len(input_shapes) == 2
        input_shape, context_shape = input_shapes
        # define shapes
        u_shape = tf.TensorShape([input_shape[-1], self.rank])
        s_shape = tf.TensorShape([self.rank])
        v_shape = tf.TensorShape([self.units, self.rank])
        w_shape = tf.TensorShape([context_shape[-1], self.rank])
        b_shape = tf.TensorShape([self.rank])
        bias_shape = tf.TensorShape([self.units])
        # define initializers
        o_initializer = tf.keras.initializers.Orthogonal()
        s_initializer = SingularValueInitializer(input_shape[-1], self.units)
        z_initializer = tf.keras.initializers.get('Zeros')
        # define variables
        self.u = self.add_weight("U", shape=u_shape, dtype=tf.float32, initializer=o_initializer)
        self.s = self.add_weight("S", shape=s_shape, dtype=tf.float32, initializer=s_initializer)
        self.v = self.add_weight("V", shape=v_shape, dtype=tf.float32, initializer=o_initializer)
        self.w = self.add_weight("W", shape=w_shape, dtype=tf.float32, initializer=o_initializer)
        self.b = self.add_weight("B", shape=b_shape, dtype=tf.float32, initializer=z_initializer)
        if self.use_bias:
            self.bias = self.add_weight("bias", shape=bias_shape, dtype=tf.float32, initializer=z_initializer)

    def __call__(self, inputs, context):
        inputs, context = inputs
        h = context @ self.w + self.b
        chi = self.context_activation(h)
        s = tf.linalg.diag(self.s) @ tf.linalg.diag(chi)
        temp = tf.einsum('nr, brr->bnr', self.u, s)
        kernel = tf.einsum('bnr,mr->bnm', temp, self.v)
        outputs = tf.einsum('bn, bnm->bm', inputs, kernel) + self.bias
        if self.use_bias:
            outputs += self.bias
        # activate
        return self.activation(outputs)


class CADense(tf.keras.layers.Layer):
    """SVD based densely connected layer."""

    def __init__(self, units: int, rank: int, activation: str = 'relu', use_bias: bool = True, name: str = 'cadense'):
        super(CADense, self).__init__()
        # initialise parameters
        self.units = units
        self.rank = rank
        self.activation = tf.keras.activations.get(activation)
        self.use_bias = use_bias

        # initialise variables
        self.u = None
        self.s = None
        self.v = None
        self.w = None
        self.bias = None
        self._name = name

    def build(self, input_shapes):
        # unpack input shapes
        assert len(input_shapes) == 2
        input_shape, context_shape = input_shapes
        # define shapes
        u_shape = tf.TensorShape([input_shape[-1], self.rank])
        s_shape = tf.TensorShape([self.rank])
        v_shape = tf.TensorShape([self.units, self.rank])
        w_shape = tf.TensorShape([context_shape[-1], self.rank])
        bias_shape = tf.TensorShape([self.units])
        # define initializers
        o_initializer = tf.keras.initializers.Orthogonal()
        s_initializer = SingularValueInitializer(input_shape[-1], self.units)
        z_initializer = tf.keras.initializers.get('Zeros')
        # define variables
        self.u = self.add_weight("U", shape=u_shape, dtype=tf.float32, initializer=o_initializer)
        self.s = self.add_weight("S", shape=s_shape, dtype=tf.float32, initializer=s_initializer)
        self.v = self.add_weight("V", shape=v_shape, dtype=tf.float32, initializer=o_initializer)
        self.w = self.add_weight("W", shape=w_shape, dtype=tf.float32, initializer=o_initializer)
        if self.use_bias:
            self.bias = self.add_weight("bias", shape=bias_shape, dtype=tf.float32, initializer=z_initializer)

    def call(self, inputs):
        data, context = inputs
        s = tf.linalg.diag(self.s + context @ self.w)
        temp = tf.einsum('nr, brr->bnr', self.u, s)
        kernel = tf.einsum('bnr,mr->bnm', temp, self.v)
        outputs = tf.einsum('bn, bnm->bm', data, kernel) + self.bias
        if self.use_bias:
            outputs += self.bias
        # activate
        return self.activation(outputs)


"""
Non context aware supplementary layers 
"""


class PositionalEncodingLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, position):
        super(PositionalEncodingLayer, self).__init__()
        self.d_model = d_model
        self.position = position
        self.positional_encoding = positional_encoding(position, d_model)

    def __call__(self, inputs, seq_len):
        seq_len = tf.shape(inputs)[1]
        inputs *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        return inputs + self.positional_encoding[:, :seq_len, :]


class MultiHeadAttention(tf.keras.models.Model):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % num_heads == 0

        self.depth = d_model // num_heads

        self.query_weights = tf.keras.layers.Dense(d_model)
        self.key_weights = tf.keras.layers.Dense(d_model)
        self.value_weights = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def __call__(self, values, keys, queries, mask):
        batch_size = tf.shape(queries)[0]

        queries = self.query_weights(queries)  # (batch_size, seq_len, d_model)
        keys = self.key_weights(keys)  # (batch_size, seq_len, d_model)
        values = self.value_weights(values)  # (batch_size, seq_len, d_model)

        queries = self.split_heads(queries, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        keys = self.split_heads(keys, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        values = self.split_heads(values, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(
            queries, keys, values, mask)

        # (batch_size, seq_len_q, num_heads, depth)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

        # (batch_size, seq_len_q, d_model)
        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))

        # (batch_size, seq_len_q, d_model)
        output = self.dense(concat_attention)
        return output, attention_weights


class AttentionBlock(tf.keras.models.Model):
    def __init__(self, d_model: int, num_heads: int, rate):
        super(AttentionBlock, self).__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.dropout = tf.keras.layers.Dropout(rate)
        self.normalization = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def __call__(self, values, keys, queries, look_ahead_mask, training):
        outputs, weights = self.attention(values, keys, queries, look_ahead_mask)
        outputs = self.dropout(outputs, training=training)
        outputs = self.normalization(queries + outputs)
        return outputs, weights


class DecoderLayer(tf.keras.models.Model):
    def __init__(self, d_model, num_heads, dff, rate: float = 0.1):
        super(DecoderLayer, self).__init__()
        self.attention_block_one = AttentionBlock(d_model, num_heads, rate)
        self.attention_block_two = AttentionBlock(d_model, num_heads, rate)

        self.final_feedforward = PointWiseFeedForward(d_model, dff)
        self.final_normalization = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.final_dropout = tf.keras.layers.Dropout(rate)

    def __call__(self, inputs, encoder_output, training, look_ahead_mask, padding_mask):
        attention_weights = {}

        inputs, weights = self.attention_block_one(
            inputs, inputs, inputs,
            look_ahead_mask, training)
        attention_weights['block_one'] = weights
        inputs, weights = self.attention_block_two(
            encoder_output, encoder_output, inputs,
            padding_mask, training)
        attention_weights['block_two'] = weights

        outputs = self.final_feedforward(inputs)  # (batch_size, input_seq_len, d_model)
        outputs = self.final_dropout(outputs, training=training)
        outputs = self.final_normalization(inputs + outputs)  # (batch_size, input_seq_len, d_model)
        return outputs, attention_weights


class EncoderLayer(tf.keras.models.Model):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()

        self.attention_block = AttentionBlock(d_model, num_heads, rate)

        self.final_feedforward = PointWiseFeedForward(d_model, dff)
        self.final_normalization = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.final_dropout = tf.keras.layers.Dropout(rate)

    def __call__(self, inputs, training, mask):
        inputs, _ = self.attention_block(inputs, inputs, inputs, mask, training)  # (batch_size, input_seq_len, d_model)

        outputs = self.final_feedforward(inputs)  # (batch_size, input_seq_len, d_model)
        outputs = self.final_dropout(outputs, training=training)
        outputs = self.final_normalization(inputs + outputs)  # (batch_size, input_seq_len, d_model)
        return outputs


class PointWiseFeedForward(tf.keras.models.Model):
    def __init__(self, d_model, width, activation: str = 'relu'):
        super(PointWiseFeedForward, self).__init__()
        # (batch_size, seq_len, dff)
        self.dense = tf.keras.layers.Dense(width, activation=activation)
        # (batch_size, seq_len, d_model)
        self.linear = tf.keras.layers.Dense(d_model)

    def __call__(self, inputs):
        inputs = self.dense(inputs)
        return self.linear(inputs)


"""
Context aware supplementary layers 
"""


class CAMultiHeadAttention(tf.keras.models.Model):
    def __init__(self, d_model: int, d_model_rank: int, num_heads: int):
        """Multi-head attention with context aware kernels

        Parameters
        ----------
        d_model
            Depth of model
        d_model_rank
            Rank of model
        num_heads
            Number of heads to split attention over
        """
        super(CAMultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_model_rank = d_model_rank

        assert d_model % num_heads == 0

        self.depth = d_model // num_heads

        self.d_model_rank = d_model_rank

        self.query_weights = CADense(d_model, d_model_rank, name='cadense_queries')
        self.key_weights = CADense(d_model, d_model_rank, name='cadense_keys')
        self.value_weights = CADense(d_model, d_model_rank, name='cadense_values')

        self.dense = CADense(d_model, d_model_rank)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def __call__(self, values, keys, queries, values_context, keys_context, queries_context, mask):
#         assert values.shape[0, 1] == values_context.shape[0, 1]
#         assert keys.shape[0, 1] == keys_context.shape[0, 1]
#         assert queries.shape[0, 1] == queries_context.shape[0, 1]
        # Necessary parameters
        batch_size, seq_len_q, depth_q = tf.shape(queries)
        _, seq_len_k, depth_k = tf.shape(keys)
        _, seq_len_v, depth_v = tf.shape(values)

        # Reshape and transform
        queries = tf.reshape(queries, (-1, depth_q))
        keys = tf.reshape(keys, (-1, depth_k))
        values = tf.reshape(values, (-1, depth_v))

        queries_context = tf.reshape(queries_context, (-1, depth_q))
        keys_context = tf.reshape(keys_context, (-1, depth_k))
        values_context = tf.reshape(values_context, (-1, depth_v))

        queries = self.query_weights([queries, queries_context])  # (batch_size * seq_len_q, d_model)
        keys = self.key_weights([keys, keys_context])  # (batch_size * seq_len_k, d_model)
        values = self.value_weights([values, values_context])  # (batch_size * seq_len_v, d_model)

        queries = tf.reshape(queries, (batch_size, seq_len_q, self.d_model))  # (batch_size, seq_len_q, d_model)
        keys = tf.reshape(keys, (batch_size, seq_len_k, self.d_model))  # (batch_size, seq_len_k, d_model)
        values = tf.reshape(values, (batch_size, seq_len_v, self.d_model))  # (batch_size, seq_len_v, d_model)

        queries = self.split_heads(queries, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        keys = self.split_heads(keys, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        values = self.split_heads(values, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(
            queries, keys, values, mask)

        # (batch_size, seq_len_q, num_heads, depth)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

        # (batch_size * seq_len_q, d_model)
        concat_attention = tf.reshape(scaled_attention, (-1, self.d_model))

        # (batch_size, seq_len_q, d_model)
        output = self.dense([concat_attention, queries_context])
        output = tf.reshape(output, (batch_size, seq_len_q, self.d_model))
        return output, attention_weights


class CAAttentionBlock(tf.keras.models.Model):
    def __init__(self, d_model: int, d_model_rank: int, num_heads: int, rate: float):
        super(CAAttentionBlock, self).__init__()
        self.attention = CAMultiHeadAttention(d_model, d_model_rank, num_heads)
        self.dropout = tf.keras.layers.Dropout(rate)
        self.normalization = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def __call__(self, values, keys, queries,
                 values_context, keys_context, queries_context,
                 look_ahead_mask, training):
        outputs, weights = self.attention(
            values, keys, queries,  # inputs
            values_context, keys_context, queries_context,  # context
            look_ahead_mask  # extra args
        )
        outputs = self.dropout(outputs, training=training)
        outputs = self.normalization(queries + outputs)
        return outputs, weights


class CAPointWiseFeedForward(tf.keras.models.Model):
    def __init__(self, d_model, d_model_rank, width, rank, activation: str = 'relu'):
        super(CAPointWiseFeedForward, self).__init__()
        self.width = width
        self.d_model = d_model
        self.d_model_rank = d_model_rank
        self.rank = rank
        self.dense = CADense(width, rank, activation=activation, name='cadense_activated')
        self.linear = CADense(d_model, d_model_rank, name='cadense_linear')

    def __call__(self, inputs: tf.Tensor, context: tf.Tensor):
#         assert inputs.shape[0, 1] == context.shape[0, 1]
        batch_size, seq_len, depth = tf.shape(inputs)
        # Reshape
        inputs = tf.reshape(inputs, (-1, depth))
        context = tf.reshape(context, (-1, depth))
        # Transform
        inputs = self.dense([inputs, context])  # (batch_size * seq_len, width)
        output = self.linear([inputs, context])  # (batch_size * seq_len, d_model)
        # Reshape
        return tf.reshape(output, (batch_size, seq_len, self.d_model))


class CAPointWiseFeedForwardBlock(tf.keras.models.Model):
    def __init__(self, d_model: int, d_model_rank: int, width: int, rank: int, rate: float):
        super(CAPointWiseFeedForwardBlock, self).__init__()
        self.feed_forward = CAPointWiseFeedForward(d_model, d_model_rank, width, rank)
        self.dropout = tf.keras.layers.Dropout(rate)
        self.normalization = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def __call__(self, inputs, context, training):
        outputs = self.feed_forward(inputs, context)
        outputs = self.dropout(outputs, training=training)
        outputs = self.normalization(inputs + outputs)
        return outputs


class CADecoderLayer(tf.keras.models.Model):
    def __init__(self, d_model, d_model_rank, num_heads, dff, dff_rank, rate: float = 0.1):
        super(CADecoderLayer, self).__init__()
        self.attention_block_one = CAAttentionBlock(d_model, d_model_rank, num_heads, rate)
        self.attention_block_two = CAAttentionBlock(d_model, d_model_rank, num_heads, rate)
        self.feedforward_block = CAPointWiseFeedForwardBlock(d_model, d_model_rank, dff, dff_rank, rate)

    def __call__(self, inputs, input_context, encoder_output, output_context,
                 training, look_ahead_mask, padding_mask):
        attention_weights = {}

        inputs, weights = self.attention_block_one(
            inputs, inputs, inputs,
            input_context, input_context, input_context,
            look_ahead_mask, training)
        attention_weights['block_one'] = weights
        inputs, weights = self.attention_block_two(
            encoder_output, encoder_output, inputs,
            output_context, output_context, input_context,
            padding_mask, training)
        attention_weights['block_two'] = weights

        outputs = self.feedforward_block(inputs, input_context, training)  # (batch_size, target_seq_len, d_model)
        return outputs, attention_weights


class CAEncoderLayer(tf.keras.models.Model):
    def __init__(self, d_model, d_model_rank, num_heads, dff, dff_rank, rate=0.1):
        super(CAEncoderLayer, self).__init__()

        self.attention_block = CAAttentionBlock(d_model, d_model_rank, num_heads, rate)
        self.feedforward_block = CAPointWiseFeedForwardBlock(d_model, d_model_rank, dff, dff_rank, rate)

    def __call__(self, inputs, context, training, mask):
        attention_weights = {}

        inputs, weights = self.attention_block(
            inputs, inputs, inputs,
            context, context, context,
            mask, training)  # (batch_size, input_seq_len, d_model)
        attention_weights['block_one'] = weights
        outputs = self.feedforward_block(inputs, context, training)  # (batch_size, input_seq_len, d_model)
        return outputs, attention_weights