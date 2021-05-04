import tensorflow as tf
from os.path import join


def phi(var, grad, nu):
    """Calculate cayley transform descent curve given a variable an it's gradient."""
    y = grad @ tf.transpose(var) - var @ tf.transpose(grad)
    return (tf.eye(y.shape[0]) - nu / 2 * y) @ (tf.eye(y.shape[0]) + nu / 2 * y)


def chi(var, grad, nu):
    """Calculate additive part of  phi = I + chi given a variable and it's gradient."""
    a = tf.concat([grad, var], axis=1)
    b = tf.concat([var, -grad], axis=1)
    skew = tf.transpose(b) @ a
    c = tf.eye(skew.shape[0]) + nu / 2 * skew
    skew_inv = tf.linalg.inv(c, adjoint=False)
    return -nu * a @ skew_inv @ tf.transpose(b)


def assembled_gradient(u, s, v, du, ds, dv, eps):
    """Calculate gradient w.r.t assembled matrix from partial gradients and variable values."""
    # Diagonal matrices for singular values
    s_matrix = tf.linalg.diag(s)
    ds_matrix = tf.linalg.diag(ds)
    # Calculate D
    s_inv = tf.linalg.diag(s ** (-1))
    d = du @ s_inv
    # Calculate A
    a = tf.where(tf.eye(ds_matrix.shape[0]) == 1., ds_matrix - tf.transpose(u) @ d, 0.0)
    # Calculate K
    i_skew = tf.ones_like(s_matrix) - tf.eye(s_matrix.shape[-1])
    k = tf.where(i_skew == 0.0, 0.0, (tf.expand_dims(s ** 2, axis=-1) - s ** 2 + eps) ** (-1))
    # Calculate B
    b = k * (tf.transpose(v) @ dv - tf.transpose(d) @ u @ s_matrix)
    # Calculate Q
    q = d + u @ (a + s_matrix @ (tf.transpose(b) + b))
    # Return dw
    return q @ tf.transpose(v)


def update_svd(u, s, v, du, ds, dv, lr_u, lr_s, lr_v, eps: float = 10e-8):
    """Update svd components such that u & v stay orthogonal and the descent corresponds to regular SGD."""
    # Calculate orthogonal update
    chi_u = chi(u, du, lr_u)
    chi_v = chi(v, dv, lr_v)
    delta_u = chi_u @ u
    delta_v = chi_v @ v
    # Calculate assembled gradient
    dw = assembled_gradient(u, s, v, du, ds, dv, eps)
    # Calculate singular value updates
    psi_u = tf.transpose(u) @ delta_u
    psi_v = tf.transpose(v) @ delta_v
    # Diagonal matrices
    s_ = tf.linalg.diag(s)
    lr_s_ = tf.linalg.diag(lr_s)
    # Diagonal part of update only using R x R matrices
    delta_s = tf.linalg.diag_part(
        psi_u@s_ + (s_ + psi_u@s_)@tf.transpose(psi_v) - lr_s_ * (tf.transpose(u + delta_u)@dw@(v + delta_v))
    )
    # Update orthogonal matrices
    u.assign_add(delta_u)
    v.assign_add(delta_v)
    # Update singular values
    s.assign_add(delta_s)


def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

    # add extra dimensions to add the padding
    # to the attention logits.
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)


def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)


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
