import tensorflow as tf


def orthogonality_number(weights: tf.Tensor) -> tf.float32:
    """Calculate kappa measure.

    Notes
    -----
    k = ||1 - W@W.T||

    Parameters
    ----------
    weights: tf.Tensor
        Orthogonal matrix (N x R)

    Returns
    -------
    tf.float32
        Orthogonality measure kappa
    """
    shape = tf.shape(weights)
    return tf.linalg.norm(tf.eye(shape[-1]) - tf.transpose(weights)@weights, axis=[-1, -2])

def conditioning_number(weights: tf.Tensor) -> tf.float32:
    """Calculate conditioning number.

    Notes
    -----
    k = S_max / S_min

    Parameters
    ----------
    weights: tf.Tensor
        Singular value vector (R)

    Returns
    -------
    tf.float32
        Conditioning number of matrix with given singular values
    """
    return tf.reduce_max(weights, axis=-1) / tf.reduce_min(weights, axis=-1)
