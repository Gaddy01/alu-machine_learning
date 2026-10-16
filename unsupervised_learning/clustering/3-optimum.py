#!/usr/bin/env python3
"""Determine the optimum number of clusters using variance."""

import numpy as np

kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """
    Determine the optimum number of clusters using variance.

    Args:
        X: Dataset of shape (n, d).
        kmin: Minimum number of clusters.
        kmax: Maximum number of clusters.
        iterations: Maximum K-means iterations.

    Returns:
        results: K-means outputs.
        d_vars: Variance differences.
        (None, None) on failure.
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None

    if not isinstance(kmin, int) or kmin <= 0:
        return None, None

    if kmax is None:
        kmax = X.shape[0]

    if not isinstance(kmax, int) or kmax <= kmin:
        return None, None

    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    results = []
    d_vars = []

    for k in range(kmin, kmax + 1):
        C, clss = kmeans(X, k, iterations)

        if C is None or clss is None:
            return None, None

        results.append((C, clss))

        if k == kmin:
            var = variance(X, C)

        d_vars.append(var - variance(X, C))

    return results, d_vars
