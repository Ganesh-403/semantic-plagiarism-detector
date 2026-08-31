"""
Utility functions for embedding vector operations
Issue: #4015
"""

import math


def l2_normalize(vector):
    """
    Normalizes a vector to have unit length (L2 norm = 1).
    """
    norm = math.sqrt(sum(val * val for val in vector))
    if norm == 0:
        return vector
    return [val / norm for val in vector]