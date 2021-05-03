import tensorflow as tf

from typing import *
from .utils import update_gradients, scaled_dot_product_attention, positional_encoding
from src.initializers import SingularValueInitializer


class SVDDense(tf.keras.layers.Layer):
    """SVD based densely connected layer."""
    def __init__(self, units: int, rank: int = None, activation: str = 'relu', use_bias: bool = True):
        """Initialise layer.

        Parameters
        ----------
        units: int
            Number of units
        rank: int
            Rank of matrix decomposition
            default value of half of number of units
        activation: str
            Activation function known in Tensorflow library
        use_bias: bool
            Bias vector boolean
        """
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
        """Build layer.

        Parameters
        ----------
        input_shape: tf.Tensorshape
            Shape of input tensor
        """
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

    def call(self, inputs: tf.Tensor):
        """Call layer.

        Parameters
        ----------
        inputs: tf.Tensor
            Input data (batch size x model depth)
        Returns
        -------
        tf.Tensor: Activated output
        """
        # build kernel
        kernel = self._u @ tf.linalg.diag(self._s) @ tf.transpose(self._v)
        # transform data
        outputs = inputs @ kernel
        # add bias
        if self.use_bias:
            outputs += self._bias
        # activate
        return self.activation(outputs)


class PositionalEncodingLayer(tf.keras.layers.Layer):
    def __init__(self, d_model: int, position: int):
        """ Initialise layer.

        Parameters
        ----------
        d_model: int
            Depth of embeddings
        position: int
            maximal position index
        """
        super(PositionalEncodingLayer, self).__init__()
        self.d_model = d_model
        self.position = position
        # Build positional encoding
        self.positional_encoding = positional_encoding(position, d_model)

    def __call__(self, inputs: tf.Tensor):
        """Call layer.

        Parameters
        ----------
        inputs: tf.Tensor
            Input data (batch size x sequence length x model depth)

        Returns
        -------
        tf.Tensor: Positional encoded input
        """
        # get sequence length
        seq_len = tf.shape(inputs)[1]
        # scale inputs
        inputs *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        # add encoding
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
