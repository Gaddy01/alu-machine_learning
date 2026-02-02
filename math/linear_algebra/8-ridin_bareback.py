#!/usr/bin/env python3
"""
Module that provides a function to perform matrix multiplication.
"""


def mat_mul(mat1, mat2):
    """
    Multiplies two 2D matrices.
    Args:
        mat1 (list of lists of int/float): First matrix
        mat2 (list of lists of int/float): Second matrix
    Returns:
        list of lists: New matrix as the result of mat1 * mat2, or None if shapes incompatible
    """
    if len(mat1[0]) != len(mat2):
        return None
    result = []
    for i in range(len(mat1)):
        row = []
        for j in range(len(mat2[0])):
            val = sum(mat1[i][k] * mat2[k][j] for k in range(len(mat2)))
            row.append(val)
        result.append(row)
    return result
