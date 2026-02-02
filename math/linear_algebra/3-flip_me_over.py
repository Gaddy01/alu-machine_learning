#!/usr/bin/env python3
"""
Module that provides a function to transpose a 2D matrix.
"""


def matrix_transpose(matrix):
    """
    Returns the transpose of a 2D matrix.
    Args:
        matrix (list of lists): The matrix to transpose
    Returns:
        list of lists: A new transposed matrix
    """
    return [list(row) for row in zip(*matrix)]
