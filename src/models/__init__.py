from typing import *

import tensorflow as tf

from src.layers import EncoderLayer, DecoderLayer, PositionalEncodingLayer
from .utils import *


class Encoder(tf.keras.models.Model):
    """Encoder for transformer model."""
    def __init__(self, num_layers: int, d_model: int, d_model_rank: Optional[int], num_heads: int, dff: int,
                 dff_rank: Optional[int], input_vocab_size: int, maximum_position_encoding: int, rate: float = 0.1):
        """

        Parameters
        ----------
        num_layers: int
            Number of encoder layers
        d_model: int
            Model depth
        d_model_rank: Optional[int]
            Model depth rank. If None regular matrices are used otherwise SVD matrices
        num_heads: int
            Number of attention heads
        dff: int
            Width of point wise feedforward layer
        dff_rank: Optional[int]
            Width rank. Idem as model depth rank.
        input_vocab_size: int
            Input vocabulary size
        maximum_position_encoding: int
            Maximal positional encoding
        rate: float
            Dropout rate
            (default is 0.1)
        """
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.d_model_rank = d_model_rank
        self.dff = dff
        self.dff_rank = dff_rank
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        self.positional_encoding = PositionalEncodingLayer(maximum_position_encoding, self.d_model)

        self.encoder_layers = [
            EncoderLayer(d_model, d_model_rank, num_heads, dff,dff_rank,  rate) for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(rate)

    def __call__(self, inputs, training, mask):
        """

        Parameters
        ----------
        inputs: tf.Tensor
            Inputs
        training: bool
            Training indicator
        mask: tf.Tensor
            Mask

        Returns
        -------
        tf.Tensor & dict
            Outputs & weights
        """
        attention_weights = {}

        # adding embedding and position encoding.
        inputs = self.embedding(inputs)  # (batch_size, input_seq_len, d_model)
        inputs = self.positional_encoding(inputs)

        inputs = self.dropout(inputs, training=training)

        for i, layer in enumerate(self.encoder_layers):
            inputs, weights = layer(inputs, training, mask)
            attention_weights[f'encoder_layer_{i}'] = weights

        return inputs, attention_weights # (batch_size, input_seq_len, d_model)


class Decoder(tf.keras.models.Model):
    """Decoder for transformer model."""
    def __init__(self, num_layers, d_model: int, d_model_rank: Optional[int], num_heads: int,
                 dff: int, dff_rank: Optional[int], target_vocab_size, maximum_position_encoding, rate: float = 0.1):
        """

        Parameters
        ----------
        num_layers: int
            Number of encoder layers
        d_model: int
            Model depth
        d_model_rank: Optional[int]
            Model depth rank. If None regular matrices are used otherwise SVD matrices
        num_heads: int
            Number of attention heads
        dff: int
            Width of point wise feedforward layer
        dff_rank: Optional[int]
            Width rank. Idem as model depth rank.
        target_vocab_size: int
            Target vocabulary size
        maximum_position_encoding: int
            Maximal positional encoding
        rate: float
            Dropout rate
            (default is 0.1)
        """
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.d_model_rank = d_model_rank
        self.dff = dff
        self.dff_rank = dff_rank
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
        self.positional_encoding = PositionalEncodingLayer(maximum_position_encoding, d_model)

        self.decoder_layers = [
            DecoderLayer(d_model, d_model_rank, num_heads, dff, dff_rank, rate) for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)

    def __call__(self, inputs, enc_output, training, look_ahead_mask, padding_mask):
        """Call decoder.

        Parameters
        ----------
        inputs: tf.Tensor
            Inputs
        enc_output: tf.Tensor
            Encoder output
        training: bool
            Training indicator
        look_ahead_mask: tf.Tensor
            Look ahead mask
        padding_mask: tf.Tensor
            Padding mask

        Returns
        -------
        tf.Tensor, dict
            Outputs, weights
        """
        attention_weights = {}

        # adding embedding and position encoding.
        inputs = self.embedding(inputs)  # (batch_size, input_seq_len, d_model)
        inputs = self.positional_encoding(inputs)

        inputs = self.dropout(inputs, training=training)

        for i, layer in enumerate(self.decoder_layers):
            inputs, weights = layer(inputs, enc_output, training, look_ahead_mask, padding_mask)

            attention_weights[f'decoder_layer{i + 1}'] = weights

        # x.shape == (batch_size, target_seq_len, d_model)
        return inputs, attention_weights


class Transformer(tf.keras.Model):
    """Sequence to sequence transformer for sequences of integers."""
    def __init__(self, num_layers: int, d_model: int, num_heads: int, dff: int, input_vocab_size: int,
                 target_vocab_size: int, input_maximum_position_encoding: int, target_maximum_position_encoding: int,
                 d_model_rank: Optional[int] = None, dff_rank: Optional[int] = None, rate: float = 0.1):
        """Transformer model initialization.
        
        Parameters
        ----------
        num_layers: int
            number of layers in encoder and decoder
        d_model: int
            depth of model
        num_heads:
            number of attention heads for MHA
        dff: int
            Units for feedforward layers after MHA
        input_vocab_size: int
            Input vocabulary size
        target_vocab_size: int
            Target vocabulary size
        input_maximum_position_encoding: int
            Maximal positional encoding for input
        target_maximum_position_encoding: int
            Maximal positional encoding for input
        d_model_rank: Optional[int]
            Rank for model depth. If not not SVD layer is used.
            (default is None)
        dff_rank: Optional[int]
            Rank for feedforward layers. If not not SVD layer is used.
            (default is None)
        rate: float
            Dropout rate
            (default is 0.1)
        """
        super(Transformer, self).__init__()

        self.encoder = Encoder(
            num_layers, d_model, d_model_rank, num_heads, dff, dff_rank,
            input_vocab_size, input_maximum_position_encoding, rate)
        self.decoder = Decoder(
            num_layers, d_model, d_model_rank, num_heads, dff, dff_rank,
            target_vocab_size, target_maximum_position_encoding, rate)

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
    def create_masks(inp: tf.Tensor, tar: tf.Tensor):
        """Create masks for transformer.

        Parameters
        ----------
        inp: tf.Tensor
            Inputs
        tar: tf.Tensor
            Targets

        Returns
        -------
        tf.Tensor, tf.Tensor, tf.Tensor:
            Encoder padding mask, Combined mask, Decoder padding mask
        """
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

    def train_step(self, inputs: tf.Tensor, targets: tf.Tensor):
        """Train step for transformer.

        Parameters
        ----------
        inputs: tf.Tensor
            Input tensor
        targets: tf.Tensor
            Target tensor

        Notes
        -----
        Calls loss function and uses optimizer so SVD compatible
        """
        # Shift targets for input and prediction
        targets_input = targets[:, :-1]
        targets_real = targets[:, 1:]

        # Create masks
        enc_padding_mask, combined_mask, dec_padding_mask = self.create_masks(inputs, targets_input)

        # Get loss & gradients
        with tf.GradientTape() as tape:
            predictions, _ = self(inputs, targets_input, True, enc_padding_mask, combined_mask, dec_padding_mask)
            loss = self.compiled_loss(targets_real, predictions)

            gradients = tape.gradient(loss, self.trainable_variables)

        # apply gradients and update loss state
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.epoch_loss.update_state(loss)

    def train(self, train_data: tf.Dataset[tf.Tensor, tf.Tensor], epochs: int, verbose: bool = True,
              save_ckpt: int = 5, ckpt_path: Optional[str] = None):
        """Train function.

        Parameters
        ----------
        train_data: tf.Dataset[tf.Tensor, tf.Tensor]
            Dataset with sequences
        epochs: int
            number of epochs
        verbose: bool
            Verbose parameter for training
        save_ckpt: int
            Amount of checkpoints to be saved
        ckpt_path: Optional[str]
            Checkpoint path. If None no saving is done.
            (default is None)
        """
        # Checkpoints manager
        if ckpt_path is not None:
            _ckpt = tf.train.Checkpoint(transformer=self, optimizer=self.optimizer)
            _ckpt_manager = tf.train.CheckpointManager(_ckpt, ckpt_path, max_to_keep=5)

        # Initialize loss state
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
            # print(f'Epoch {epoch}, train metrics: {train_metrics:.4f}' + ', '.join(['train_metrics']))
            # print(f'Epoch {epoch}, validation metrics: {validation_metrics:.4f}' + ', '.join(['validation_metrics']))

            if ((epoch + 1) % save_ckpt == 0) & (ckpt_path is not None):
                ckpt_save_path = _ckpt_manager.save()
                print('Saving checkpoint for epoch {} at {}'.format(epoch + 1, ckpt_save_path))

    def restore(self, ckpt_path):
        """

        Parameters
        ----------
        ckpt_path

        Returns
        -------

        """
        _ckpt = tf.train.Checkpoint(transformer=self, optimizer=self.optimizer)
        _ckpt_manager = tf.train.CheckpointManager(_ckpt, ckpt_path, max_to_keep=5)
        # if a checkpoint exists, restore the latest checkpoint.
        if _ckpt_manager.latest_checkpoint:
            _ckpt.restore(_ckpt_manager.latest_checkpoint)
            print('Latest checkpoint restored!!')

    def predict(self, data, max_length: int = 40, return_weights: bool = False):
        """

        Parameters
        ----------
        data
        max_length
        return_weights

        Returns
        -------

        """
        # Iterate over batches
        for encoder_input in data:
            output = self.predict_batch(encoder_input, max_length, return_weights)
            yield output

    def predict_batch(self, encoder_input, max_length: int = 40, return_weights: bool = False, start_tokens=None):
        """

        Parameters
        ----------
        encoder_input
        max_length
        return_weights
        start_tokens

        Returns
        -------

        """
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
