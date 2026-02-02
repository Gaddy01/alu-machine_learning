#!/usr/bin/env python3
"""
Module that provides a function to compute the shape of a matrix.
"""
def matrix_shape(matrix):
    """
    Calculates the shape of a matrix.
    Args:
        matrix (list): A list representing a matrix (possibly multi-dimensional)
    Returns:
        list: A list of integers representing the shape of the matrix
    """
    shape = []
    while isinstance(matrix, list):
        shape.append(len(matrix))
        if len(matrix) == 0:
            break
        matrix = matrix[0]
    return shape
