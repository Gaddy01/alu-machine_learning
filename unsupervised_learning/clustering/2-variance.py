
#!/usr/bin/env python3
"""Calculate total intra-cluster variance."""

import numpy as np


def variance(X, C):
    """Calculate total intra-cluster variance."""
    if not isinstance(X, np.ndarray):
        return None

    if not isinstance(C, np.ndarray):
        return None

    if X.ndim != 2 or C.ndim != 2:
        return None

    if X.shape[0] == 0 or C.shape[0] == 0:
        return None

    if X.shape[1] == 0 or X.shape[1] != C.shape[1]:
        return None

    if not (np.issubdtype(X.dtype, np.number)
            and np.issubdtype(C.dtype, np.number)):
        return None

    distances = np.sum((X - C[:, np.newaxis]) ** 2, axis=-1)
    min_distances = np.min(distances, axis=0)

    return np.sum(min_distances)
