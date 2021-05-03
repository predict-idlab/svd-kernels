import tensorflow as tf


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
    # Diagonal part of update only using R x R matrices
    delta_s = tf.linalg.diag_part(
        psi_u@s_ + (s_ + psi_u@s_)@tf.transpose(psi_v) - lr_s * (tf.transpose(u + delta_u)@dw@(v + delta_v))
    )
    return delta_u, delta_s, delta_v
