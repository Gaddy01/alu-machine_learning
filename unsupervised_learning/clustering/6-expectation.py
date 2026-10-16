#!/usr/bin/env python3
"""Calculate the expectation step of the EM algorithm."""

import numpy as np

pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """
    Calculate the expectation step of a GMM.

    Args:
        X: Dataset with shape (n, d).
        pi: Prior probabilities with shape (k,).
        m: Cluster means with shape (k, d).
        S: Covariance matrices with shape (k, d, d).

    Returns:
        g: Posterior probabilities with shape (k, n).
        l: Total log-likelihood.
        (None, None) on failure.
    """
    if not isinstance(X, np.ndarray):
        return None, None

    if not isinstance(pi, np.ndarray):
        return None, None

    if not isinstance(m, np.ndarray):
        return None, None

    if not isinstance(S, np.ndarray):
        return None, None

    if X.ndim != 2 or pi.ndim != 1:
        return None, None

    if m.ndim != 2 or S.ndim != 3:
        return None, None

    n, d = X.shape
    k = pi.shape[0]

    if n == 0 or d == 0 or k == 0:
        return None, None

    if m.shape != (k, d) or S.shape != (k, d, d):
        return None, None

    if not all(np.issubdtype(a.dtype, np.number)
               for a in (X, pi, m, S)):
        return None, None

    if not np.all(np.isfinite(pi)):
        return None, None

    if np.any(pi < 0) or not np.isclose(np.sum(pi), 1):
        return None, None

    g = np.zeros((k, n))

    for i in range(k):
        p = pdf(X, m[i], S[i])

        if p is None:
            return None, None

        g[i] = pi[i] * p

    total = np.sum(g, axis=0)

    if np.any(total <= 0) or not np.all(np.isfinite(total)):
        return None, None

    l = np.sum(np.log(total))

    g = g / total

    return g, l
