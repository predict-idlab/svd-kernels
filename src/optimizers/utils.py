import tensorflow as tf
from os.path import join
from typing import Union


def phi(var: tf.Tensor, grad: tf.tensor, nu: Union[float, tf.Tensor]) -> tf.Tensor:
    """Calculate Cayley transform descent curve given a variable an it's gradient.

    Parameters
    ----------
    var: tf.Tensor
        Variable values tensor
    grad: tf.Tensor
        Variable gradients tensor
    nu: tf.Tensor or float
        Learning rate on manifold
    Returns
    -------
        Cayley transformed gradient curve
    """
    # Calculate asymmetric gradient
    w = grad @ tf.transpose(var) - var @ tf.transpose(grad)
    # Get unit matrix
    i = tf.eye(w.shape[0])
    # Scale with learning rate
    w *= tf.divide(nu, 2.)
    # Calculate Cayley transform
    return tf.linalg.inv(i - w) @ (i + w)


def chi(var: tf.Tensor, grad: tf.tensor, nu: Union[float, tf.Tensor]) -> tf.Tensor:
    """Calculate additive part of  phi = I + chi given a variable and it's gradient.

    Parameters
    ----------
    var: tf.Tensor
        Variable values tensor
    grad: tf.Tensor
        Variable gradients tensor
    nu: tf.Tensor or float
        Learning rate on manifold
    Returns
    -------
        Woodbury Morrison formulae for chi
    """
    # Get 2R x N parts for calculation
    a = tf.concat([grad, var], axis=1)
    b = tf.concat([var, -grad], axis=1)
    # Calculate skew matrix
    skew = tf.transpose(b) @ a
    skew *= tf.divide(nu, 2.)
    skew += tf.eye(skew.shape[0])
    # Calculate inverse
    skew_inv = tf.linalg.inv(skew, adjoint=False)
    # Calculate chi
    return -nu * a @ skew_inv @ tf.transpose(b)


def assembled_gradient(
        u: tf.Tensor, s: tf.Tensor, v: tf.Tensor,
        du: tf.Tensor, ds: tf.Tensor, dv: tf.Tensor,
        eps: float = 10e-8) -> tf.Tensor:
    """Calculate gradient w.r.t assembled matrix from partial gradients and variable values.

    Parameters
    ----------
    u: tf.Tensor
        Left orthogonal matrix (N x R)
    s: tf.Tensor
        Singular values vector (R)
    v: tf.Tensor
        Right orthogonal matrix (M x R)
    du: tf.Tensor
        Left orthogonal matrix gradients (N x R)
    ds: tf.Tensor
        Singular values vector gradients (R)
    dv: tf.Tensor
        Right orthogonal matrix gradients (M x R)
    eps: float
        Epsilon for numerical stability of division and roots
        (default is 10e-8)

    Returns
    -------
        Gradient w.r.t. assembled matrix
    """
    # Diagonal matrices for singular values
    s_matrix = tf.linalg.diag(s)
    ds_matrix = tf.linalg.diag(ds)
    # Calculate D
    s_inv = tf.linalg.diag(tf.power(s, -1))
    d = du @ s_inv
    # Calculate A
    a = tf.where(tf.eye(ds_matrix.shape[0]) == 1., ds_matrix - tf.transpose(u) @ d, 0.0)
    # Calculate K
    i_skew = tf.ones_like(s_matrix) - tf.eye(s_matrix.shape[-1])
    k = tf.where(i_skew == 0.0, 0.0, (tf.expand_dims(tf.power(s, 2), axis=-1) - tf.power(s, 2) + eps) ** (-1))
    # Calculate B
    b = k * (tf.transpose(v) @ dv - tf.transpose(d) @ u @ s_matrix)
    # Calculate Q
    q = d + u @ (a + s_matrix @ (tf.transpose(b) + b))
    # Return dw
    return q @ tf.transpose(v)


def update_svd(u, s, v, du, ds, dv, lr_u, lr_s, lr_v, eps: float = 10e-8):
    """Update svd components such that u & v stay orthogonal and the descent corresponds to regular SGD.

    Parameters
    ----------
    Parameters
    ----------
    u: tf.Tensor
        Left orthogonal matrix (N x R)
    s: tf.Tensor
        Singular values vector (R)
    v: tf.Tensor
        Right orthogonal matrix (M x R)
    du: tf.Tensor
        Left orthogonal matrix gradients (N x R)
    ds: tf.Tensor
        Singular values vector gradients (R)
    dv: tf.Tensor
        Right orthogonal matrix gradients (M x R)
    lr_u
    lr_s
    lr_v
    eps: float
        Epsilon for numerical stability of division and roots
        (default is 10e-8)

    Returns
    -------

    """
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
