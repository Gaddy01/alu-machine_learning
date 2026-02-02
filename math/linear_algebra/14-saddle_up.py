#!/usr/bin/env python3
"""
Module that provides a function to perform matrix multiplication using NumPy.
"""

import numpy as np


def np_matmul(mat1, mat2):
    """
    Multiplies two numpy arrays using matrix multiplication.
    Args:
        mat1 (numpy.ndarray): First matrix
        mat2 (numpy.ndarray): Second matrix
    Returns:
        numpy.ndarray: Result of mat1 @ mat2
    """
    return np.matmul(mat1, mat2)
