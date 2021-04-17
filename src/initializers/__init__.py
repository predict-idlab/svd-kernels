import tensorflow as tf
import numpy as np

from .utils import singular_value_probability


class SingularValueInitializer(tf.keras.initializers.Initializer):
    """Initializer for S kernel of src based layer."""
    def __init__(self, n, m):
        """Initialise initializer.

        Parameters
        ----------
        n: int
            First dimension
        m: int
            Second dimension
        """
        super(SingularValueInitializer, self).__init__()
        self.n = n
        self.m = m

    def __call__(self, shape: tf.TensorShape, dtype=tf.float32):
        """Call initializer.

        Parameters
        ----------
        shape: tf.Tensorshape
            Tensorshape of singular value vector (Rx1)
        dtype: dtype
            Data type for initializer
            (standard tf.float32)
        Returns
        -------
            Randomly sampled vector of singular values from corresponding distribution.
        """
        variance = 2/self.n
        t = np.min([self.n, self.m])
        r = np.max([self.n, self.m])
        grit = int(10e5)  # enough spacing for fine grained singular values
        l_min = 10e-12  # min is small so no 0 singular values
        l_max = 6*r - 2*t + 2  # maximal zero of laguerre polynomials
        x = np.linspace(l_min, l_max, grit)
        p_s = singular_value_probability(x, r, t)
        return np.sqrt(np.random.choice(x * variance, size=shape[0], p=p_s / np.sum(p_s)))
