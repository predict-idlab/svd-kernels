import numpy as np
from scipy.special import factorial as fac


def log_factorial(k: int, threshold: int = 64):
    """Log factorial function that resorts to stirling formula.

    Parameters
    ----------
    k: int
        integer to factorial
    threshold: int
        threshold from which to start using stirling formula

    Returns
    -------
        Natural logarithm of k factorial [ln(k!)]
    """
    if k < threshold:
        return np.log(fac(k))
    else:
        return k * (np.log(k) - 1)


def orthonormal_laguerre(x: np.array, n: int, k: int, phi_0: np.array, phi_1: np.array):
    """Recursive function for orthonormal laguerre functions.

    Parameters
    ----------
    x: np.array
        x values
    n: int
        First parameter of associated laguerre polynomials
    k: int
        Second parameter of associated laguerre polynomials
    phi_0: np.array
        phi(n-1)
    phi_1: np.array
        phi(n)
    Returns
    -------
        phi(n + 1)
    """
    c_0 = (2 * n + 1 + k - x) / ((n + k + 1) * (n + 1)) ** (1 / 2)
    c_1 = (n * (n + k) / ((n + k + 1) * (n + 1))) ** (1 / 2)
    return c_0 * phi_0 - c_1 * phi_1


def singular_value_probability(x: np.array, r: int, t: int):
    """Probability distribution of singular values of random gaussian matrix.

    Parameters
    ----------
    x: np.array
        singular value values
    r: int
        Maximal dimension of matrix
    t: int
        Minimal dimension of matrix

    Returns
    -------
        Probability distribution over singular values
    """
    k = r - t
    polynomials = []
    phi_0 = np.exp(log_factorial(k) * (-1 / 2) - x / 2 + np.log(x) * (k / 2))
    polynomials.append(phi_0)
    phi_1 = np.exp(log_factorial(k + 1) * (-1 / 2) - x / 2 + np.log(x) * (k / 2)) * (1 + k - x)
    polynomials.append(phi_1)
    for i in range(1, t - 1):
        phi_i = orthonormal_laguerre(x, i, k, polynomials[i], polynomials[i - 1])
        polynomials.append(phi_i)
    p_s = (1 / t) * np.sum(np.array(polynomials) ** 2, axis=0)
    return p_s

