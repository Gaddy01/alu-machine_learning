#!/usr/bin/env python3
"""
Module that provides a function to perform 
element-wise arithmetic operations on numpy arrays.
"""


def np_elementwise(mat1, mat2):
    """
    Performs element-wise addition, subtraction,
    multiplication, and division.
    Args:
        mat1 (numpy.ndarray): First array or matrix
        mat2 (numpy.ndarray or scalar): Second array/matrix or scalar
    Returns:
        tuple: (sum, difference, product, quotient)
    """
    return (mat1 + mat2, mat1 - mat2, mat1 * mat2, mat1 / mat2)

