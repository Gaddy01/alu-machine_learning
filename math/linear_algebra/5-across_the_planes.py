#!/usr/bin/env python3
"""
Module that provides a function to add two 2D matrices element-wise.
"""


def add_matrices2D(mat1, mat2):
    """
    Adds two 2D matrices element-wise.
    Args:
        mat1 (list of lists of int/float): First matrix
        mat2 (list of lists of int/float): Second matrix
    Returns:
        list of lists: New matrix with element-wise sums, or None if shapes differ
    """
    if len(mat1) != len(mat2) or any(len(r1) != len(r2) for r1, r2 in zip(mat1, mat2)):
        return None
    return [[a + b for a, b in zip(r1, r2)] for r1, r2 in zip(mat1, mat2)]
