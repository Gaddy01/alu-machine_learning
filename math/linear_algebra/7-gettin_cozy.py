#!/usr/bin/env python3
"""
Module that provides a function to concatenate
two 2D matrices along a specified axis.
"""


def cat_matrices2D(mat1, mat2, axis=0):
    """
    Concatenates two 2D matrices along the given axis.
    Args:
        mat1 (list of lists of int/float): First matrix
        mat2 (list of lists of int/float): Second matrix
        axis (int): Axis along which to concatenate (0 = rows, 1 = columns)
    Returns:
        list of lists: New concatenated matrix, or None if shapes incompatible
    """
    if axis == 0:
        if any(len(row) != len(mat1[0]) for row in mat1 + mat2):
            return None
        return [row[:] for row in mat1] + [row[:] for row in mat2]
    elif axis == 1:
        if len(mat1) != len(mat2):
            return None
        return [r1[:] + r2[:] for r1, r2 in zip(mat1, mat2)]
    else:
        return None

