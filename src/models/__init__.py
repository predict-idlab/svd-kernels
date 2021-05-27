from typing import *

import tensorflow as tf

from os.path import join
from numpy import delete
from src.layers import EncoderLayer, DecoderLayer, PositionalEncodingLayer
from .utils import *


class Encoder(tf.keras.models.Model):
    def __init__(self, num_layers: int, d_model: int, num_heads: int, dff: int,
                 input_vocab_size: int, maximum_position_encoding: int, rate: float = 0.1):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        self.positional_encoding = PositionalEncodingLayer(maximum_position_encoding, self.d_model)

        self.encoder_layers = [EncoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(rate)

    def __call__(self, inputs, training, mask):
        seq_len = tf.shape(inputs)[1]

        # adding embedding and position encoding.
        inputs = self.embedding(inputs)  # (batch_size, input_seq_len, d_model)
        inputs = self.positional_encoding(inputs, seq_len)

        inputs = self.dropout(inputs, training=training)

        for layer in self.encoder_layers:
            inputs = layer(inputs, training, mask)

        return inputs  # (batch_size, input_seq_len, d_model)


class Decoder(tf.keras.models.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size,
                 maximum_position_encoding, rate=0.1):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
        self.positional_encoding = PositionalEncodingLayer(maximum_position_encoding, d_model)

        self.decoder_layers = [DecoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)

    def __call__(self, inputs, enc_output, training, look_ahead_mask, padding_mask):
        attention_weights = {}
        seq_len = tf.shape(inputs)[1]

        # adding embedding and position encoding.
        inputs = self.embedding(inputs)  # (batch_size, input_seq_len, d_model)
        inputs = self.positional_encoding(inputs, seq_len)

        inputs = self.dropout(inputs, training=training)

        for i, layer in enumerate(self.decoder_layers):
            inputs, weights = layer(inputs, enc_output, training, look_ahead_mask, padding_mask)

            attention_weights[f'decoder_layer{i + 1}'] = weights

        # x.shape == (batch_size, target_seq_len, d_model)
        return inputs, attention_weights


class Transformer(tf.keras.Model):
    def __init__(self, num_layers: int, d_model: int, num_heads: int, dff: int,
                 input_vocab_size: int, target_vocab_size: int, input_maximum_position_encoding: int,
                 target_maximum_position_encoding: int, rate: float = 0.1):
        super(Transformer, self).__init__()

        self.encoder = Encoder(
            num_layers, d_model, num_heads, dff, input_vocab_size, input_maximum_position_encoding, rate)
        self.decoder = Decoder(
            num_layers, d_model, num_heads, dff, target_vocab_size, target_maximum_position_encoding, rate)

        self.input_vocab_size = input_vocab_size
        self.target_vocab_size = target_vocab_size

        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def __call__(self, inputs, targets, training, enc_padding_mask, look_ahead_mask, dec_padding_mask):
        # (batch_size, inp_seq_len, d_model)
        encoded_inputs = self.encoder(inputs, training, enc_padding_mask)
        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        decoded_inputs, weights = self.decoder(targets, encoded_inputs, training, look_ahead_mask, dec_padding_mask)
        # (batch_size, tar_seq_len, target_vocab_size)
        final_output = self.final_layer(decoded_inputs)

        return final_output, weights

    @staticmethod
    def create_masks(inp, tar):
        # Encoder padding mask
        enc_padding_mask = create_padding_mask(inp)

        # Used in the 2nd attention block in the decoder.
        # This padding mask is used to mask the encoder outputs.
        dec_padding_mask = create_padding_mask(inp)

        # Used in the 1st attention block in the decoder.
        # It is used to pad and mask future tokens in the input received by
        # the decoder.
        look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
        dec_target_padding_mask = create_padding_mask(tar)
        combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

        return enc_padding_mask, combined_mask, dec_padding_mask

    def train_step(self, inputs, targets):
        targets_input = targets[:, :-1]
        targets_real = targets[:, 1:]

        enc_padding_mask, combined_mask, dec_padding_mask = self.create_masks(inputs, targets_input)

        with tf.GradientTape() as tape:
            predictions, _ = self(inputs, targets_input, True, enc_padding_mask, combined_mask, dec_padding_mask)
            loss = self.compiled_loss(targets_real, predictions)

            gradients = tape.gradient(loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.epoch_loss.update_state(loss)

    def train(self, train_data, epochs, verbose: bool = True, save_ckpt: int = 5, ckpt_path: Optional[str] = None):
        # Checkpoints manager
        if ckpt_path is not None:
            _ckpt = tf.train.Checkpoint(transformer=self, optimizer=self.optimizer)
            _ckpt_manager = tf.train.CheckpointManager(_ckpt, ckpt_path, max_to_keep=5)

        self.epoch_loss = tf.keras.metrics.Mean()

        # Training
        for epoch in range(epochs):
            self.epoch_loss.reset_states()
            # training step
            for (batch, (inputs, targets)) in enumerate(train_data):
                if (batch + 1) % 100 == 0:
                    print(f'Batch: {batch}, loss: {self.epoch_loss.result()}')
                self.train_step(inputs, targets)

            if verbose:
                print(f'Epoch {epoch}, Loss: {self.epoch_loss.result()}')
            #                 print(f'Epoch {epoch}, train metrics: {train_metrics:.4f}' + ', '.join(['train_metrics']))
            #                 print(f'Epoch {epoch}, validation metrics: {validation_metrics:.4f}' + ', '.join(['validation_metrics']))

            if ((epoch + 1) % save_ckpt == 0) & (ckpt_path is not None):
                ckpt_save_path = _ckpt_manager.save()
                print('Saving checkpoint for epoch {} at {}'.format(epoch + 1, ckpt_save_path))

    def restore(self, ckpt_path):
        _ckpt = tf.train.Checkpoint(transformer=self, optimizer=self.optimizer)
        _ckpt_manager = tf.train.CheckpointManager(_ckpt, ckpt_path, max_to_keep=5)
        # if a checkpoint exists, restore the latest checkpoint.
        if _ckpt_manager.latest_checkpoint:
            _ckpt.restore(_ckpt_manager.latest_checkpoint)
            print('Latest checkpoint restored!!')

    def predict(self, data, max_length: int = 40, return_weights: bool = False):
        # Iterate over batches
        for encoder_input in data:
            output = self.predict_batch(encoder_input, max_length, return_weights)
            yield output

    def predict_batch(self, encoder_input, max_length: int = 40, return_weights: bool = False, start_tokens=None):
        # Batch size and all indices
        batch_size = encoder_input.shape[0]
        indices = tf.range(batch_size)
        # Starting tokens
        if start_tokens is None:
            start_tokens = tf.tile([[self.target_vocab_size - 2]], [batch_size, 1])
        # Empty output
        empty = tf.zeros((batch_size, max_length), dtype=tf.int32)
        output = tf.concat([start_tokens, empty], axis=-1)
        # Iterate over maximum length
        for i in range(max_length):
            n_indices = indices.shape[0]
            # Select indices that haven't finished
            encoder_input = tf.gather(encoder_input, indices, axis=0)
            decoder_input = tf.slice(tf.gather(output, indices, axis=0), [0, 0], [n_indices, i + 1])
            # Masking
            enc_padding_mask, combined_mask, dec_padding_mask = self.create_masks(encoder_input, decoder_input)
            # Predictions
            predictions, attention_weights = self(encoder_input,
                                                  decoder_input,
                                                  False,
                                                  enc_padding_mask,
                                                  combined_mask,
                                                  dec_padding_mask)

            # Select last tokens from sequence dimension
            predictions = tf.squeeze(predictions[:, -1:, :])  # (batch_size, vocab_size)
            # Predict most likely token
            predicted_ids = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
            # Get indices where sentence has not ended
            mask = tf.math.not_equal(predicted_ids, self.target_vocab_size - 1)
            indices = tf.boolean_mask(indices, mask)
            # Get position in sequence
            positions = tf.repeat([i + 1], indices.shape)
            indices_nd_positions = tf.stack([indices, positions], axis=-1)
            # Add predictions to output
            output = tf.tensor_scatter_nd_update(output, indices_nd_positions, tf.boolean_mask(predicted_ids, mask))
            if n_indices == 0: break
        return output[:, 1:], attention_weights if return_weights else output[:, 1:]


"""
Context aware feed forward
"""

# class CAFeedForward(tf.keras.models.Model):


"""
Context aware transformer
"""


class CAEncoder(tf.keras.models.Model):
    def __init__(self, num_layers: int, d_model: int, d_model_rank: int, num_heads: int, dff: int, dff_rank: int,
                 maximum_position_encoding: int, input_vocab_size: Optional[int] = None,
                 context_vocab_size: Optional[int] = None, rate: float = 0.1):
        """Build context-aware encoder.

        Parameters
        ----------
        num_layers: int
            Number of layers in encoder
        d_model: int
            Depth of model
        d_model_rank
            Rank of model
        num_heads: int
            Number of heads
        dff: int
            Width of feedforward layers
        dff_rank: int
            Rank of feedforward layers
        maximum_position_encoding: int
            Maximal positioning encoding size
        input_vocab_size: Optional[int]
            Input vocabulary size
        context_vocab_size: Optional[int]
            Context vocabulary size
        rate: float
            dropout rate (default 0.1)
        """
        super(CAEncoder, self).__init__()

        self.num_layers = num_layers
        self.d_model = d_model
        self.d_model_rank = d_model_rank
        self.dff = dff
        self.dff_rank = dff_rank

        self.input_vocab_size = input_vocab_size
        self.context_vocab_size = context_vocab_size

        # Embedding layer for input and context if needed
        if input_vocab_size is not None:
            self.input_embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        if context_vocab_size is not None:
            self.context_embedding = tf.keras.layers.Embedding(context_vocab_size, d_model)

        self.positional_encoding = PositionalEncodingLayer(maximum_position_encoding, self.d_model)

        self.encoder_layers = [
            CAEncoderLayer(d_model, d_model_rank, num_heads, dff, dff_rank, rate) for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(rate)

    def __call__(self, inputs, context, training, mask):
        attention_weights = {}
        seq_len = inputs.shape[1]

        # adding embedding and position encoding.
        if self.input_vocab_size is not None:
            inputs = self.input_embedding(inputs)  # (batch_size, input_seq_len, d_model)
        if self.context_vocab_size is not None:
            context = self.context_embedding(context)  # (batch_size, input_seq_len, d_model)

        inputs = self.positional_encoding(inputs, seq_len)
        inputs = self.dropout(inputs, training=training)

        for idx, layer in enumerate(self.encoder_layers):
            inputs, weights = layer(inputs, context, training, mask)
            attention_weights[f'encoder_layer_{idx}'] = weights
        return inputs, context, attention_weights  # (batch_size, input_seq_len, d_model)


class CADecoder(tf.keras.models.Model):
    def __init__(self, num_layers: int, d_model: int, d_model_rank: int, num_heads: int, dff: int, dff_rank: int,
                 maximum_position_encoding: int, input_vocab_size: Optional[int] = None,
                 context_vocab_size: Optional[int] = None, rate: float = 0.1):
        """Build context-aware decoder.

        Parameters
        ----------
        num_layers: int
            Number of layers in encoder
        d_model: int
            Depth of model
        d_model_rank
            Rank of model
        num_heads: int
            Number of heads
        dff: int
            Width of feedforward layers
        dff_rank: int
            Rank of feedforward layers
        maximum_position_encoding: int
            Maximal positioning encoding size
        input_vocab_size: Optional[int]
            Input vocabulary size
        context_vocab_size: Optional[int]
            Context vocabulary size
        rate: float
            dropout rate (default 0.1)
        """
        super(CADecoder, self).__init__()

        self.num_layers = num_layers
        self.d_model = d_model
        self.d_model_rank = d_model_rank
        self.dff = dff
        self.dff_rank = dff_rank

        self.input_vocab_size = input_vocab_size
        self.context_vocab_size = context_vocab_size

        # Embedding layer for input and context if needed
        if input_vocab_size is not None:
            self.input_embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        if context_vocab_size is not None:
            self.context_embedding = tf.keras.layers.Embedding(context_vocab_size, d_model)

        self.positional_encoding = PositionalEncodingLayer(maximum_position_encoding, self.d_model)

        self.decoder_layers = [
            CADecoderLayer(d_model, d_model_rank, num_heads, dff, dff_rank, rate) for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(rate)

    def __call__(self, inputs, input_context, enc_output, output_context, training, look_ahead_mask, padding_mask):
        attention_weights = {}
        seq_len = inputs.shape[1]

        # adding embedding and position encoding.
        if self.input_vocab_size is not None:
            inputs = self.input_embedding(inputs)  # (batch_size, input_seq_len, d_model)
        if self.context_vocab_size is not None:
            input_context = self.context_embedding(input_context)  # (batch_size, input_seq_len, d_model)

        inputs = self.positional_encoding(inputs, seq_len)

        inputs = self.dropout(inputs, training=training)

        for idx, layer in enumerate(self.decoder_layers):
            inputs, weights = layer(inputs, input_context, enc_output, output_context,
                                    training, look_ahead_mask, padding_mask)
            attention_weights[f'decoder_layer_{idx}'] = weights
        return inputs, attention_weights  # (batch_size, input_seq_len, d_model)


class CATransformer(tf.keras.Model):
    def __init__(self, num_layers: int, d_model: int, d_model_rank: int, num_heads: int,
                 dff: int, dff_rank: int, input_vocab_size: Optional[int], target_vocab_size: Optional[int],
                 input_maximum_position_encoding: int, target_maximum_position_encoding: int,
                 input_context_vocab_size: int = None, target_context_vocab_size: int = None, rate: float = 0.1):
        """Build transformer with context aware kernels.

        Parameters
        ----------
        num_layers: int
            Number of encoder and decoder layers
        d_model: int
            Depth of model for attention heads
        d_model_rank: int
            Rank of model for attention heads
        num_heads: int
            Number of attention heads
        dff: int
            Units for feedforward layers
        dff_rank: int
            Rank for feedforward layers
        input_vocab_size: int
            Input vocabulary size for embedding
        target_vocab_size: int
            Target vocabulary size for embedding
        input_maximum_position_encoding: int
            Positional encoding size for input
        target_maximum_position_encoding: int
            Positional encoding size for target
        input_context_vocab_size: int
            Input vocabulary size for context embedding
        target_context_vocab_size: int
            Target vocabulary size for context embedding
        rate: float
            Dropout rate (default 0.1)
        """
        super(CATransformer, self).__init__()

        # Model parameters
        self.num_layers = num_layers
        self.d_model = d_model
        self.d_model_rank = d_model_rank
        self.num_heads = num_heads
        self.dff = dff
        self.dff_rank = dff_rank

        # Vocabulary sizes
        self.input_vocab_size = input_vocab_size
        self.target_vocab_size = target_vocab_size

        self.input_context_vocab_size = input_context_vocab_size
        self.target_context_vocab_size = target_context_vocab_size

        # Positional encoding sizes
        self.input_maximum_position_encoding = input_maximum_position_encoding
        self.target_maximum_position_encoding = target_maximum_position_encoding

        # Dropout rate
        self.rate = rate

        # Encoder & decoder
        self.encoder = CAEncoder(num_layers, d_model, d_model_rank, num_heads, dff, dff_rank,
                                 input_maximum_position_encoding,  input_vocab_size, input_context_vocab_size, rate)
        self.decoder = CADecoder(num_layers, d_model, d_model_rank, num_heads, dff, dff_rank,
                                 target_maximum_position_encoding, target_vocab_size, target_context_vocab_size, rate)

        # Projection layer
        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def __call__(self, inputs, input_context, targets, target_context,
                 training, enc_padding_mask, look_ahead_mask, dec_padding_mask, return_weights: bool = True):
        """Call transformer.

        Parameters
        ----------
        inputs: tf.Tensor[int]
            Input data (batch_size, seq_len_input,)
        input_context: tf.Tensor[int] or tf.Tensor([float]
            Input context (batch_size, seq_len_input, ) or (batch_size, seq_len_input, d_model)
        targets: tf.Tensor[int]
            Target data (batch_size, seq_len_target, )
        target_context: tf.Tensor[int] or tf.Tensor([float]
            Target context (batch_size, seq_len_target, ) or (batch_size, seq_len_target, d_model)
        training: bool
            Training indicator
        enc_padding_mask: tf.Tensor[bool]
            Padding for encoder
        look_ahead_mask: tf.Tensor[bool]
            Look ahead mask
        dec_padding_mask: tf.Tensor[bool]
            Padding mask for decoder
        return_weights: bool
            Whether to return weights (default True)
        Returns
        -------
        Output: tf.Tensor[int]
            Output of transformer
        Encoder & decoder weights: dict[str->tf.Tensor[float]]
            Weights for encoder and decoder
        """
        # (batch_size, inp_seq_len, d_model)
        encoded_inputs, encoded_context, encoder_weights = self.encoder(inputs, input_context, training,
                                                                        enc_padding_mask)
        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        decoded_inputs, decoder_weights = self.decoder(targets, target_context, encoded_inputs, encoded_context,
                                                       training, look_ahead_mask, dec_padding_mask)
        # (batch_size, tar_seq_len, target_vocab_size)
        final_output = self.final_layer(decoded_inputs)
        weights = [encoder_weights, decoder_weights]
        if return_weights:
            return final_output, weights
        else:
            return final_output

    @staticmethod
    def create_masks(inp, tar):
        # Encoder padding mask
        enc_padding_mask = create_padding_mask(inp)

        # Used in the 2nd attention block in the decoder.
        # This padding mask is used to mask the encoder outputs.
        dec_padding_mask = create_padding_mask(inp)

        # Used in the 1st attention block in the decoder.
        # It is used to pad and mask future tokens in the input received by
        # the decoder.
        look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
        dec_target_padding_mask = create_padding_mask(tar)
        combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

        return enc_padding_mask, combined_mask, dec_padding_mask

    def get_layer_updates(self, gradients, variables, names, target_context, input_context):
        # Get gradients and variables for components
        u, s, v, w = variables
        du, ds, dv, dw = gradients
        # Calculate orthogonal update
        chi_u = chi(u, du, self.optimizer.learning_rate)
        chi_v = chi(v, dv, self.optimizer.learning_rate)
        u_update = u + chi_u @ u
        v_update = v + chi_v @ v
        # Context updated coefficients depending on encoder or decoder
        position = names[1]
        if (('decoder' in position) and ('values' in position)) or (('decoder' in position) and ('keys' in position)):
            context = target_context  # if 'decoder' and ('values' or 'keys') in layer.name else input_context
            if self.target_context_vocab_size is not None:
                context = self.decoder.context_embedding(context)
        else:
            context = input_context
            if self.input_context_vocab_size is not None:
                context = self.encoder.context_embedding(context)
        context = tf.reshape(context, (-1, context.shape[-1]))
        s_ = s + context @ w
        # calculate assembled gradient
        dk = batch_assembled_gradient(u, s_, v, du, ds, dv)
        # Calculate singular value updates
        psi_u = tf.transpose(u) @ chi_u @ u
        psi_v = tf.transpose(v) @ chi_v @ v
        s_matrix = tf.linalg.diag(s_)
        s_update_matrix = psi_u @ s_matrix + (s_matrix + psi_u @ s_matrix) @ tf.transpose(
            psi_v) - self.optimizer.learning_rate * (
                                  tf.transpose(u_update) @ dk @ v_update + tf.linalg.diag(context @ dw)
                          )
        w_update = self.optimizer.learning_rate * dw
        return u_update, v_update, s_update_matrix, w_update

    def distributed_train_step(self, strategy, inputs, input_context, targets, target_context):
        per_replica_losses = strategy.run(self.train_step, args=(inputs, input_context, targets, target_context))
        return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

    def train_step(self, inputs, input_context, targets, target_context):
        # Select input targets and real targets
        target_inputs = targets[:, :-1]
        target_context_inputs = target_context[:, :-1]
        targets_real = targets[:, 1:]
        # Make padding masks
        enc_padding_mask, combined_mask, dec_padding_mask = self.create_masks(inputs, target_inputs)

        with tf.GradientTape() as tape:
            # Predictions and loss w.r.t target inputs and real targets
            predictions, _ = self(inputs, input_context, target_inputs, target_context_inputs,
                                  True, enc_padding_mask, combined_mask, dec_padding_mask)
            loss = self.compiled_loss(targets_real, predictions)
            # Variables
            names, variables = list(self.unpacked.keys()), list(self.unpacked.values())
            # Gradients
            gradients = tape.gradient(loss, variables)
        # Indices of svd variables
        slices = [slice(idx, idx + 4) for idx, name in enumerate(names) if ('cadense' in name) & ('U' in name)]
        length = range(len(variables))
        svd_indices = [idx for indices in slices for idx in length[indices]]
        for indices in slices:
            # Get gradients and variables for components
            u, s, v, w = variables[indices]
            u_update, v_update, s_update_matrix, w_update = self.get_layer_updates(
                gradients[indices], variables[indices], names[indices], target_context, input_context)
            # Update orthogonal matrices
            u.assign_add(u_update)
            v.assign_add(v_update)
            # Update singular values
            s.assign_add(tf.reduce_mean(tf.linalg.diag_part(s_update_matrix), axis=0))
            # regular updates
            w.assign_sub(w_update)
            # Optimize other variables
        remainder = zip(delete(gradients, svd_indices), delete(variables, svd_indices))
        self.optimizer.apply_gradients(remainder)
        self.epoch_loss.update_state(tf.reduce_mean(loss))

    def train(self, train_data, epochs, strategy=None, validation_data=None,
              train_metrics=None, validation_metrics=None, verbose: bool = True, save_ckpt: int = 5,
              ckpt_path: Optional[str] = None):
        # Checkpoints manager
        if ckpt_path is not None:
            _ckpt = tf.train.Checkpoint(transformer=self, optimizer=self.optimizer)
            _ckpt_manager = tf.train.CheckpointManager(_ckpt, ckpt_path, max_to_keep=5)

        self.epoch_loss = tf.keras.metrics.Mean()
        self.unpacked = {
            join(name, var.name): var for name, layer in unpack([self]) for var in layer.trainable_variables}

        # Training
        for epoch in range(epochs):
            self.epoch_loss.reset_states()
            # training step
            for (batch, (inputs, input_context, targets, target_context)) in enumerate(train_data):
                if batch % 100 == 0:
                    print(f'Batch: {batch}, Loss: {self.epoch_loss.result()}')
                if strategy is not None:
                    self.distributed_train_step(strategy, inputs, input_context, targets, target_context)
                else:
                    self.train_step(inputs, input_context, targets, target_context)

            if verbose:
                print(f'Epoch {epoch}, Loss: {self.epoch_loss.result()}')
            # print(f'Epoch {epoch}, train metrics: {train_metrics:.4f}' + ', '.join(['train_metrics']))
            # print(f'Epoch {epoch}, validation metrics: {validation_metrics:.4f}' + ', '.join(['validation_metrics']))

            if ((epoch + 1) % save_ckpt == 0) & (ckpt_path is not None):
                ckpt_save_path = _ckpt_manager.save()
                print('Saving checkpoint for epoch {} at {}'.format(epoch + 1, ckpt_save_path))

    def distributed_test_step(self, dataset_inputs):
        return self.strategy.run(self.test_step, args=(dataset_inputs,))

    def test_step(self, encoder_input, encoder_context, decoder_context, max_length: int = 40,
                      return_weights: bool = False):
        # Batch size and all indices
        batch_size = encoder_input.shape[0]
        indices = tf.range(batch_size)
        # Starting tokens
        start_tokens = tf.tile([[self.target_vocab_size - 2]], [batch_size, 1])
        # Empty output
        empty = tf.zeros((batch_size, max_length), dtype=tf.int32)
        output = tf.concat([start_tokens, empty], axis=-1)
        # Iterate over maximum length
        for i in range(max_length):
            n_indices = indices.shape[0]
            # Select indices that haven't finished
            encoder_input = tf.gather(encoder_input, indices, axis=0)
            decoder_input = tf.slice(tf.gather(output, indices, axis=0), [0, 0], [n_indices, i + 1])
            # Masking
            enc_padding_mask, combined_mask, dec_padding_mask = self.create_masks(encoder_input, decoder_input)
            # Predictions
            predictions, attention_weights = self(encoder_input,
                                                  encoder_context,
                                                  decoder_input,
                                                  decoder_context,
                                                  False,
                                                  enc_padding_mask,
                                                  combined_mask,
                                                  dec_padding_mask)

            # Select last tokens from sequence dimension
            predictions = tf.squeeze(predictions[:, -1:, :])  # (batch_size, vocab_size)
            # Predict most likely token
            predicted_ids = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
            # Get indices where sentence has not ended
            mask = tf.math.not_equal(predicted_ids, self.target_vocab_size - 1)
            indices = tf.boolean_mask(indices, mask)
            # Get position in sequence
            positions = tf.repeat([i + 1], indices.shape)
            indices_nd_positions = tf.stack([indices, positions], axis=-1)
            # Add predictions to output
            output = tf.tensor_scatter_nd_update(output, indices_nd_positions, tf.boolean_mask(predicted_ids, mask))
            if n_indices == 0: break
        return output[:, 1:], attention_weights if return_weights else output[:, 1:]

    def test(self, data, max_length: int = 40, return_weights: bool = False):
        # Iterate over batches
        for encoder_input in data:
            output = self.predict_batch(encoder_input, max_length, return_weights)
            yield output

    def restore(self, ckpt_path):
        _ckpt = tf.train.Checkpoint(transformer=self, optimizer=self.optimizer)
        _ckpt_manager = tf.train.CheckpointManager(_ckpt, ckpt_path, max_to_keep=5)
        # if a checkpoint exists, restore the latest checkpoint.
        if _ckpt_manager.latest_checkpoint:
            _ckpt.restore(_ckpt_manager.latest_checkpoint)
            print('Latest checkpoint restored!!')