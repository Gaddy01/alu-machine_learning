#!/usr/bin/env python3
"""
Module that provides a function to perform element-wise
arithmetic operations on numpy arrays.
"""

import numpy as np


def np_elementwise(mat1, mat2):
    """
    Performs element-wise addition, subtraction,
    multiplication, and division.
    Args:
        mat1 (array-like): First array or matrix
        mat2 (array-like or scalar): Second array/matrix or scalar
    Returns:
        tuple: (sum, difference, product, quotient)
        of element-wise operations
    """
    mat1 = np.array(mat1)
    mat2 = np.array(mat2)
    return (mat1 + mat2, mat1 - mat2, mat1 * mat2, mat1 / mat2)

