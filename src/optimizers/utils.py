import tensorflow as tf

from os.path import join
from typing import Union, Iterable, List, Optional


def phi(var: tf.Tensor, grad: tf.Tensor, nu: Union[float, tf.Tensor], k: Optional[int] = None) -> tf.Tensor:
    """Calculate Cayley transform descent curve given a variable an it's gradient.
    Parameters
    ----------
    var: tf.Tensor
        Variable values tensor
    grad: tf.Tensor
        Variable gradients tensor
    nu: tf.Tensor or float
        Learning rate on manifold
    k: Optional[int]
        Number of iterations for fixed-point iteration calculation
        (default is None)
    Returns
    -------
        Cayley transformed gradient curve
    """
    # Calculate asymmetric gradient
    w = grad @ tf.transpose(var) - var @ tf.transpose(grad)
    # Scale with learning rate
    w *= tf.divide(nu, 2.)
    # Get unit matrix
    i = tf.eye(w.shape[0])
    # Calculate Cayley with inverse when no iterations are defined
    if k is None:
        y = tf.linalg.inv(i + w) @ (i - w)
    else:
        assert k > -1
        # Initialize y
        y = i - w  
        # Calculate iteratively
        for _ in range(k):    
            y = i - w@(i + y)
    return y


def chi(var: tf.Tensor, grad: tf.Tensor, nu: float, k: Optional[int] = None) -> tf.Tensor:
    """Calculate additive part of  phi = I + chi given a variable and it's gradient.
    Notes
    -----
    Nu can't be tf.Tensor due to shape in calculation of skew part.
    This is why adaptive learning rate are pressed into gradient calculation.
    See SVDAdam for more info.

    Parameters
    ----------
    var: tf.Tensor
        Variable values tensor
    grad: tf.Tensor
        Variable gradients tensor
    nu: float
        Learning rate on manifold
    k: Optional[int]
        Number of iterations for fixed-point iteration calculation
        (default is None)
    Returns
    -------
        Woodbury Morrison formulae for chi
    """
    if k is None:
        # Get 2R x N parts for calculation
        a = tf.concat([grad, var], axis=1)
        b = tf.concat([var, -grad], axis=1)
        # Calculate skew matrix
        skew = tf.transpose(b) @ a
        skew *= tf.divide(nu, 2.)
        skew += tf.eye(skew.shape[0])
        # Calculate inverse
        skew_inv = tf.linalg.inv(skew)
        # Calculate chi
        y = - nu * a @ skew_inv @ tf.transpose(b)
    else:
        assert k > -1
        # Calculate asymmetric gradient
        w = grad @ tf.transpose(var) - var @ tf.transpose(grad)
        # Scale with learning rate
        w *= tf.divide(nu, 2.)
        # Get unit matrix
        i = tf.eye(w.shape[0])
        # Initialize chi
        y = - w
        # Calculate iteratively
        for _ in range(k):
            y = - w @ (2*i + y)
    return y


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
    s_inv = tf.linalg.diag(tf.math.pow(s, -1))
    d = du @ s_inv
    # Calculate A
    a = tf.where(tf.eye(ds_matrix.shape[0]) == 1., ds_matrix - tf.transpose(u) @ d, 0.0)
    # Calculate K
    i_skew = tf.ones_like(s_matrix) - tf.eye(s_matrix.shape[-1])
    k = tf.where(i_skew == 0.0, 0.0, (tf.expand_dims(tf.math.pow(s, 2), axis=-1) - tf.math.pow(s, 2) + eps) ** (-1))
    # Calculate B
    b = k * (tf.transpose(v) @ dv - tf.transpose(d) @ u @ s_matrix)
    # Calculate Q
    q = d + u @ (a + s_matrix @ (tf.transpose(b) + b))
    # Return dw
    return q @ tf.transpose(v)


def update_svd(u, s, v, du, ds, dv, lr_u, lr_s, lr_v, eps: float = 10e-8, method: str = 'chi', k: Optional[int] = None):
    """Update svd components such that u & v stay orthogonal and the descent corresponds to regular SGD.
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
    lr_u: float
        Left orthogonal matrix learning rate
    lr_s: float
        Singular values learning rate
    lr_v: float
        Right orthogonal matrix learning rate
    eps: float
        Epsilon for numerical stability of division and roots
        (default is 10e-8)
    method: str
        String indicating method to calculate cayley transform
        (default is 'chi')
    k: Optional[int]
        Number of iterations for fixed-point iteration calculation
        (default is None)
    Returns
    -----
    Gradients updates for U, S & V variables based on gradients and learning rates.
    """
    # Calculate orthogonal update
    if method == 'chi':
        chi_u = chi(u, du, lr_u, k)
        chi_v = chi(v, dv, lr_v, k)
    elif method == 'phi':
        chi_u = phi(u, du, lr_u, k) - tf.eye(u.shape[0])
        chi_v = phi(v, dv, lr_v, k) - tf.eye(v.shape[0])
    else:
        raise ValueError('Incorrect method for Cayley transform calculation. Needs to be either "phi" or "chi".')
    # Calculate update step
    delta_u = chi_u @ u
    delta_v = chi_v @ v
    # Calculate assembled gradient
    dw = assembled_gradient(u, s, v, du, ds, dv, eps)
    # Calculate singular value updates
    psi_u = tf.transpose(u) @ delta_u
    psi_v = tf.transpose(v) @ delta_v
    s_ = tf.linalg.diag(s)
    # Diagonal part of update only using R x R matrices or vectors
    delta_s = tf.linalg.diag_part(psi_u@s_ + (s_ + psi_u@s_)@tf.transpose(psi_v) - lr_s * (tf.transpose(u + delta_u)@dw@(v + delta_v)))
    return delta_u, delta_s, delta_v


def unpack(packed: Iterable) -> List:
    """Unpack a model architecture.
    Notes
    -----
    This method unpacks in the same order as model.layers in Tensorflow.
    Parameters
    ----------
    packed: Iterable
        Iterable containing item(s) with '.layer' attribute
    Returns
    -------
        Zipped list containing names and layer items
    """
    # Initialize lists
    unpacked = []
    names = []
    # Enumerate architecture
    for elements in packed:
        # Unpack layers
        if hasattr(elements, 'layers'):
            # Recurrent part
            for name, element in unpack(elements.layers):
                name = join(elements.name, name)
                unpacked.append(element)
                names.append(name)
        # add to list and end recurrence
        else:
            unpacked.append(elements)
            names.append(elements.name)
    # return list of unpacked outputs
    return list(zip(names, unpacked))