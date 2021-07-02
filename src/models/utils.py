import tensorflow as tf
from os.path import join
import numpy as np

def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

    # add extra dimensions to add the padding
    # to the attention logits.
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)


def numpy_look_ahead(size):
    mask = 1.0 - np.triu(np.ones((size, size)), 0)
    return mask.astype(np.float32)

@tf.function
def create_look_ahead_mask(size):
    mask = tf.numpy_function(numpy_look_ahead, [size], tf.float32)
    return mask

##  Doesn't work because numpy calculation in backend
## https://github.com/tensorflow/models/issues/9706#issuecomment-791113516
# def create_look_ahead_mask(size):
#     mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
#     return mask  # (seq_len, seq_len)


def unpack(packed):
    unpacked = []
    names = []
    for elements in packed:
        if hasattr(elements, 'layers'):
            for name, element in unpack(elements.layers):
                name = join(elements.name, name)
                unpacked.append(element)
                names.append(name)
        else:
            unpacked.append(elements)
            names.append(elements.name)
    return list(zip(names, unpacked))


def batch_transpose(a):
    return tf.transpose(a, [0, 2, 1])


def batch_mul(a, b, transpose_a=False, transpose_b=False):
    if transpose_a:
        a = batch_transpose(a)
    if transpose_b:
        b = batch_transpose(b)
    return tf.einsum('bik,bkj->bij', a, b)


def batch_assembled_gradient(u, s, v, du, ds, dv, eps = 10e-8):
    # Diagonal matrices for singular values
    s_matrix = tf.linalg.diag(s)
    ds_matrix = tf.linalg.diag(ds)
    # Calculate D
    s_inv = tf.linalg.diag((s + eps)**(-1))
    d = du@s_inv
    # Calculate A
    i = tf.eye(ds_matrix.shape[-1], batch_shape=[s.shape[0]])
    a = tf.where(i == 1., ds_matrix - tf.transpose(u)@d, 0.0)
    # Calculate K
    i_skew = tf.ones_like(s_matrix) - i
    k = tf.where(i_skew == 0.0, 0.0,  (tf.expand_dims(s ** 2, axis=-1) - tf.expand_dims(s, axis=-2) ** 2 + eps) ** (-1))
    # Calculate B
    b = k * (tf.transpose(v)@dv - batch_mul(d, u@s_matrix, True, False))
    # Calculate Q
    q = d + u@(a + batch_mul(s_matrix, batch_transpose(b) + b, True, False))
    # Return dw
    return q@tf.transpose(v)
