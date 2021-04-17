import tensorflow as tf
from src.models import Transformer


if __name__ == '__main__':
    parameters = {
        'num_layers': 2,
        'd_model': 512,
        'num_heads': 8,
        'width': 2048,
        'input_vocab_size': 8500,
        'target_vocab_size': 8000,
        'pe_input': 10000,
        'pe_target': 6000
    }
    
    transformer = Transformer(**parameters)
    
    source = tf.random.uniform((64, 38), dtype=tf.int64, minval=0, maxval=200)
    target = tf.random.uniform((64, 36), dtype=tf.int64, minval=0, maxval=200)
    
    settings = {
        'training': False,
        'enc_padding_mask': None,
        'look_ahead_mask': None,
        'dec_padding_mask': None
    }
    
    output, _ = transformer(source, target, **settings)

    # (batch_size, tar_seq_len, target_vocab_size)
    output_shape = output.shape
    print(output_shape)
