import tensorflow as tf

from typing import *
from .utils import update_gradients, scaled_dot_product_attention, positional_encoding
from src.initializers import SingularValueInitializer


class SVDDense(tf.keras.layers.Layer):
    """SVD based densely connected layer."""
    def __init__(self, units: int, rank: Optional[int] = None, activation: str = 'relu', use_bias: bool = True):
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

    def __call__(self, inputs: tf.Tensor):
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
    """Multihead attention with SVD layers"""
    def __init__(self, d_model, d_model_rank, num_heads):
        """Initialize layer

        Parameters
        ----------
        d_model: int
            Model feature dimension depth
        d_model_rank
            Model feature dimension rank
        num_heads: int
            Number of attention heads
        """
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_model_rank = d_model_rank

        assert d_model % num_heads == 0

        self.depth = d_model // num_heads

        self.query_weights = SVDDense(d_model, d_model_rank)
        self.key_weights = SVDDense(d_model, d_model_rank)
        self.value_weights = SVDDense(d_model, d_model_rank)

        self.dense = SVDDense(d_model, d_model_rank)

    def split_heads(self, x: tf.Tensor, batch_size: int):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)

        Parameters
        ----------
        x: tf.Tensor
            Inputs for splitting
        batch_size: int
            Batch size

        Returns
        -------
        Split inputs
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def __call__(self, values: tf.Tensor, keys: tf.Tensor, queries: tf.Tensor, mask: tf.Tensor):
        """Call multihead attention.

        Parameters
        ----------
        values: tf.Tensor[float]
            Values for attention
        keys: tf.Tensor[float]
            Keys for attention
        queries: tf.Tensor[float]
            Queries for attention
        mask: tf.Tensor[bool]
            Attention mask
        Returns
        -------
        Attention output and attention weights
        """
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
    """Attention block composition layer"""
    def __init__(self, d_model: int, d_model_rank: int, num_heads: int, rate, return_weights: bool = False):
        """Initialize layer.

        Parameters
        ----------
        d_model: int
            Model feature dimension
        d_model_rank: int
            Model feature rank
        num_heads: int
            Number of heads
        rate: float
            Dropout rate
        return_weights: bool
            Whether to return weights
            (default to False)
        """
        super(AttentionBlock, self).__init__()
        # Initialize layers
        self.attention = MultiHeadAttention(d_model, d_model_rank, num_heads)
        self.dropout = tf.keras.layers.Dropout(rate)
        self.normalization = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        # initialize parameters
        self.return_Weights = return_weights

    def __call__(self, values: tf.Tensor, keys: tf.Tensor, queries: tf.Tensor,
                 look_ahead_mask: tf.Tensor, training: bool):
        """Call layer.

        Parameters
        ----------
        values: tf.Tensor[float]
            Values for attention
        keys: tf.Tensor[float]
            Keys for attention
        queries: tf.Tensor[float]
            Queries for attention
        look_ahead_mask: tf.Tensor[bool]
            Look ahead mask for time series
        training: bool
            training indicator
        Returns
        -------
        Outputs and/or weights based on indicator
        """
        # Attend inputs
        outputs, weights = self.attention(values, keys, queries, look_ahead_mask)
        # Dropout
        outputs = self.dropout(outputs, training=training)
        # Layer normalization
        outputs = self.normalization(queries + outputs)
        # Return outputs
        return outputs, weights if self.return_weights else outputs


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


class LSTM(tf.keras.models.Model):
    """LSTM cell with SVD kernels."""
    def __init__(self, units: int, rank: int, activation: str = 'tanh', recurrent_activation: str = 'sigmoid',
                 use_bias: bool = True, unit_forget_bias: bool = True, dropout: float = 0.0,
                 recurrent_dropout: float = 0.0, return_sequences: bool = False,
                 return_state: bool = False, go_backwards: bool = False):
        """Initialize LSTM layer

        Parameters
        ----------
        units:  int
            Hidden size.
        rank: int
            Rank of hidden matrices.
        activation: str
            Activation function for output.
            (default is tanh)
        recurrent_activation: str
            Activation function for recurrent part.
            (default is sigmoid)
        use_bias: bool
            Bias indicator for output.
            (default is True)
        unit_forget_bias:
            Bias indicator for forget gate.
            (default is true)
        dropout: float
            Dropout rate for output.
            (default is 0.0)
        recurrent_dropout: float
            Dropout rate for recurrent part.
            (default is 0.0)
        return_sequences: bool
            Whether to return full output sequence.
            If states are returned this also controls whether or not full state sequences are returned.
            (default is False)
        return_state: bool
            Whether to return state. Returns context and hidden state if True.
            (default is False)
        go_backwards: bool
            Whether to run through time component in reverse.
            (default is False)
        """
        super(LSTM, self).__init__()
        # Regular parameters
        self.units = units
        self.rank = rank
        self.use_bias = use_bias
        self.activation = activation
        self.dropout = dropout
        # Recurrent parameters
        self.recurrent_activation = recurrent_activation
        self.recurrent_dropout = recurrent_dropout
        # Forget parameters
        self.unit_forget_bias = unit_forget_bias
        # Calculation parameters
        self.return_sequences = return_sequences
        self.return_state = return_state
        self.go_backwards = go_backwards
        # Forget gate components
        self.linear_forget_w1 = SVDDense(self.units, self.rank, use_bias=self.unit_forget_bias)
        self.linear_forget_r1 = SVDDense(self.units, self.rank, use_bias=False)
        self.sigmoid_forget = tf.keras.activations.get(self.recurrent_activation)
        # Input gate components
        self.linear_gate_w2 = SVDDense(self.units, self.rank, use_bias=self.use_bias)
        self.linear_gate_r2 = SVDDense(self.units, self.rank, use_bias=False)
        self.sigmoid_gate = tf.keras.activations.get(self.recurrent_activation)
        # Cell memory components
        self.linear_gate_w3 = SVDDense(self.units, self.rank, use_bias=self.use_bias)
        self.linear_gate_r3 = SVDDense(self.units, self.rank, use_bias=False)
        self.activation_gate = tf.keras.activations.get(self.activation)
        # Out gate components
        self.linear_gate_w4 = SVDDense(self.units, self.rank, use_bias=self.use_bias)
        self.linear_gate_r4 = SVDDense(self.units, self.rank, use_bias=False)
        self.sigmoid_hidden_out = tf.keras.activations.get(self.recurrent_activation)
        self.activation_final = tf.keras.activations.get(self.activation)

    def forget(self, x: tf.Tensor, h: tf.Tensor):
        """Forget gate.

        Parameters
        ----------
        x: tf.Tensor
            Input
        h: tf.Tensor
            Hidden state

        Returns
        -------
            Activated forget equation
        """
        x = self.linear_forget_w1(x)
        h = self.linear_forget_r1(h)
        return self.sigmoid_forget(x + h)

    def input_gate(self, x: tf.Tensor, h: tf.Tensor):
        """Forget gate.

        Parameters
        ----------
        x: tf.Tensor
            Input
        h: tf.Tensor
            Hidden state

        Returns
        -------
        Activated input
        """
        # Equation 1. input gate
        x = self.linear_gate_w2(x)
        h = self.linear_gate_r2(h)
        return self.sigmoid_gate(x + h)

    def cell_memory_gate(self, i: tf.Tensor, f: tf.Tensor, x: tf.Tensor, h: tf.Tensor, c: tf.Tensor):
        """Memory gate.

        Parameters
        ----------
        i: tf.Tensor
            Gated input
        f: tf.Tensor
            Forget gate
        x: tf.Tensor
            Inputs
        h: tf.Tensor
            Hidden state
        c: tf.Tensor
            Context state

        Returns
        -------
        Next context state
        """
        x = self.linear_gate_w3(x)
        h = self.linear_gate_r3(h)
        # new information part that will be injected in the new context
        k = self.activation_gate(x + h)
        g = k * i
        # forget old context/cell info
        c *= f
        # learn new context/cell info
        c += g
        return c

    def out_gate(self, x: tf.Tensor, h: tf.Tensor):
        """Output gate.

        Parameters
        ----------
        x: tf.tensor
            Inputs
        h: tf.Tensor
            Hidden state

        Returns
        -------
        Activated output
        """
        x = self.linear_gate_w4(x)
        h = self.linear_gate_r4(h)
        return self.sigmoid_hidden_out(x + h)

    def __call__(self, inputs: tf.Tensor, mask: Optional[tf.Tensor] = None,
                 training: Optional[bool] = None, initial_state: Optional[List[tf.Tensor]] = None):
        """Call layer.

        Parameters
        ----------
        inputs:
            A 3D tensor with shape [batch, timesteps, feature].
        mask:
            Binary tensor of shape [batch, timesteps] indicating whether a given timestep should be masked
            (optional, defaults to None).
        training:
            Python boolean indicating whether the layer should behave in training mode or in inference mode.
            This argument is passed to the cell when calling it.
            This is only relevant if dropout or recurrent_dropout is used (optional, defaults to None).
        initial_state:
            List of initial state tensors to be passed to the first call of the cell
            (optional, defaults to None which causes creation of zero-filled initial state tensors).

        Returns
        -------
        List of output tensors based on configuration
        """
        # Unstack inputs
        inputs = tf.unstack(inputs, axis=-2)
        # Initialize outputs
        outputs = tf.zeros_like(inputs[0])[..., tf.newaxis, ...]
        # Initialize states
        h, c = tf.zeros_like(inputs[0]), tf.zeros_like(inputs[0]) if initial_state is None else initial_state
        # Get sequence and reverse if necessary
        sequence = reversed(inputs) if self.go_backwards else inputs
        # Calculate outputs
        for x in sequence:
            # Equation 1. input gate
            i = self.input_gate(x, h)
            # Equation 2. forget gate
            f = self.forget(x, h)
            # Equation 3. updating the cell memory
            c = self.cell_memory_gate(i, f, x, h, c)
            # Equation 4. calculate the main output gate
            o = self.out_gate(x, h)
            # Equation 5. produce next hidden output
            h = o * self.activation_final(c)
            # Stack
            outputs = tf.concat([outputs, o], axis=-2)
        # Return sequence or last output
        out = outputs if self.return_sequences else outputs[:, -1, :]
        # Return states or output
        out = [out, c, h] if self.return_state else out
        return out
