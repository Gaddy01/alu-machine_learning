#!/usr/bin/env python3
"""Perform expectation maximization for a GMM."""

import numpy as np

initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k,
                             iterations=1000, tol=1e-5, verbose=False):
    """
    Perform expectation maximization for a GMM.

    Args:
        X: Dataset of shape (n, d).
        k: Number of clusters.
        iterations: Maximum number of iterations.
        tol: Convergence tolerance.
        verbose: Whether to print progress.

    Returns:
        pi: Cluster prior probabilities.
        m: Cluster means.
        S: Covariance matrices.
        g: Posterior probabilities.
        log_likelihood: Final log-likelihood.
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None, None, None

    if X.shape[0] == 0 or X.shape[1] == 0:
        return None, None, None, None, None

    if not isinstance(k, int) or k <= 0:
        return None, None, None, None, None

    if not isinstance(iterations, int) or iterations <= 0:
        return None, None, None, None, None

    if not isinstance(tol, float) or tol < 0:
        return None, None, None, None, None

    if not isinstance(verbose, bool):
        return None, None, None, None, None

    pi, m, S = initialize(X, k)

    if pi is None or m is None or S is None:
        return None, None, None, None, None

    g, log_likelihood = expectation(X, pi, m, S)

    if g is None or log_likelihood is None:
        return None, None, None, None, None

    msg = "Log Likelihood after {} iterations: {}"

    if verbose:
        print(msg.format(0, round(log_likelihood, 5)))

    for i in range(iterations):
        prev_like = log_likelihood

        pi, m, S = maximization(X, g)

        if pi is None or m is None or S is None:
            return None, None, None, None, None

        g, log_likelihood = expectation(X, pi, m, S)

        if g is None or log_likelihood is None:
            return None, None, None, None, None

        current_iteration = i + 1

        if verbose and current_iteration % 10 == 0:
            print(msg.format(current_iteration,
                             round(log_likelihood, 5)))

        if abs(prev_like - log_likelihood) <= tol:
            break

    if verbose and current_iteration % 10 != 0:
        print(msg.format(current_iteration,
                         round(log_likelihood, 5)))

    return pi, m, S, g, log_likelihood
