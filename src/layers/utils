import tensorflow as tf
import numpy as np


def calculate_chi(x, g, nu, return_step: bool = True):
    """Calculate chi according to SMW formula [1]

    Parameters
    ----------
    x: tf.Tensor
        Stieffel manifold matrix (NxR)
    g: tf.Tensor
        derivative of loss (L) w.r.t X (NxR)
    nu: float
        Update increment for step on manifold
    return_step: bool
        whether to return chi with negative step size (nu) or without

    Returns
    -------
    chi: tf.Tensor
        Update step defined by PHI = I + CHI [1] or I - nu * CHI_
    """
    a = tf.concat([g, x], axis=1)
    b = tf.concat([x, -g], axis=1)
    skew = tf.transpose(b)@a
    c = tf.eye(skew.shape[0]) + nu/2 * skew
    skew_inv = tf.linalg.inv(c, adjoint=False)
    chi = a @skew_inv@tf.transpose(b)
    return -nu * chi if return_step else chi


@tf.custom_gradient
def update_gradients(u, s, v, nu, learning_rate):
    """Update gradients for src layer.

    Notes
    -----
    Derivation and working see notebook + paper
    S is a vector here

    References
    ----------
    [1] Notes on stieffel manifold optimization
    [2] Structured layers using matrix backpropagation

    Parameters
    ----------
    u: tf.Tensor
        Orthogonal matrix (M x R)
    s: tf.Tensor
        singular values (R)
    v: tf.Tensor
        Orthogonal matrix (N x R)
    nu: float
        Stieffel manifold update step
    learning_rate: float
        Learning rate

    Returns
    -------
    u, s, v, grad_fn: tf.Tensor x3, function
        u, s, v and function generating updated gradients
    """
    s_matrix = tf.linalg.diag(s)

    def grad_fn(du, ds, dv, variables=None):
        """Gradient function.

        Parameters
        ----------
        du: tf.Tensor
            gradient tensor dL/du
        ds: tf.Tensor
            gradient tensor dL/ds
        dv: tf.Tensor
            gradient tensor dL/dv
        Returns
        -------
        Gradients
        """
        # define eps
        eps = 10e-8
        # Unpack
        ds_matrix = tf.linalg.diag(ds)
        # Calculate stieffel manifold update [1]
        chi_u = calculate_chi(u, du, nu)
        chi_v = calculate_chi(v, dv, nu)
        phi_u = tf.eye(chi_u.shape[0]) + chi_u
        phi_v = tf.eye(chi_v.shape[0]) + chi_v
        du_updated = -chi_u @ u
        dv_updated = -chi_v @ v
        # Calculate weight update vector from partial derivatives [2]
        f = tf.transpose((tf.expand_dims(s ** 2, axis=-1) - s ** 2 + eps) ** (-1))
        i_skew = tf.ones_like(s_matrix) - tf.eye(s_matrix.shape[0])
        k = tf.transpose(tf.where(i_skew == 0, 0, f))
        s_inv = tf.linalg.diag(s ** (-1))
        d = du @ s_inv
        m = tf.transpose(k) * (tf.transpose(v) @ (dv - v @ tf.transpose(d) @ u @ s_matrix))
        # Calculate ds*
        dw = tf.reduce_sum(
            [
                d @ tf.transpose(v),
                u @ (tf.where(i_skew == 0, ds_matrix - tf.transpose(u) @ d, 0)) @ tf.transpose(v),
                u @ s_matrix @ (tf.transpose(m) + m) @ tf.transpose(v)
            ], axis=0
        )
        ds_s = tf.reduce_sum(
            [
                s_matrix @ tf.transpose(v) @ chi_v @ v,
                tf.transpose(chi_u @ u) @ u @ s_matrix,
                tf.transpose(chi_u @ u) @ u @ s_matrix @ tf.transpose(v) @ chi_v @ v
            ], axis=0
        )
        ds_w = tf.transpose(phi_u @ u) @ dw @ (phi_v @ v)
        ds_updated = tf.linalg.diag_part(-(ds_s/learning_rate * ds_w))
        return du_updated, ds_updated, dv_updated, 0, 0

    return [tf.identity(u), tf.identity(s), tf.identity(v)], grad_fn


def scaled_dot_product_attention(q, k, v, mask):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead)
    but it must be broadcastable for addition.

    Args:
    q: query shape == (..., seq_len_q, depth)
    k: key shape == (..., seq_len_k, depth)
    v: value shape == (..., seq_len_v, depth_v)
    mask: Float tensor with shape broadcastable
          to (..., seq_len_q, seq_len_k). Defaults to None.

    Returns:
    output, attention_weights
    """

    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights


def positional_encoding(position, d_model):
    angle_rates = 1 / np.power(10000, (2 * (np.arange(d_model)[np.newaxis, :] // 2.)) / d_model)
    angle_rads = np.arange(position)[:, np.newaxis] * angle_rates

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    encoding = angle_rads[tf.newaxis, ...]
    return tf.cast(tf.transpose(encoding, [0, 2, 1]), dtype=tf.float32)
