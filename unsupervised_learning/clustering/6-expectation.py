#!/usr/bin/env python3
"""Calculate the expectation step of the EM algorithm."""

import numpy as np

pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """Calculate posterior probabilities and log-likelihood."""

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

    if not np.issubdtype(X.dtype, np.number):
        return None, None

    if not np.issubdtype(pi.dtype, np.number):
        return None, None

    if not np.issubdtype(m.dtype, np.number):
        return None, None

    if not np.issubdtype(S.dtype, np.number):
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

    log_likelihood = np.sum(np.log(total))
    g = g / total

    return g, log_likelihood
