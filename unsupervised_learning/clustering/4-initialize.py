#!/usr/bin/env python3
"""Initialize Gaussian Mixture Model parameters."""

import numpy as np

kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    """Initialize GMM priors, means, and covariance matrices."""

    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None

    if X.shape[0] == 0 or X.shape[1] == 0:
        return None, None, None

    if not isinstance(k, int) or k <= 0 or k > X.shape[0]:
        return None, None, None

    n, d = X.shape

    # Equal prior probabilities
    pi = np.full(k, 1 / k)

    # Initialize means using K-means
    m, _ = kmeans(X, k)

    if m is None:
        return None, None, None

    # Identity covariance matrix per cluster
    S = np.tile(np.eye(d), (k, 1, 1))

    return pi, m, S
