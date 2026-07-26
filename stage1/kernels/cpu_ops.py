"""
CPU implementations of Stage 1 kernels using NumPy.
These serve as:
1. Reference implementations for correctness testing
2. Baseline for benchmarking against GPU versions
"""

import numpy as np


def vector_add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Element-wise addition of two vectors.

    Args:
        a: First input array
        b: Second input array (must be same shape as a)

    Returns:
        Element-wise sum a + b
    """
    return a + b


def matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Matrix multiplication.

    Args:
        a: First matrix of shape (M, K)
        b: Second matrix of shape (K, N)

    Returns:
        Product matrix of shape (M, N)
    """
    return np.matmul(a, b)


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Softmax function: exp(x) / sum(exp(x)) along specified axis.

    Uses the numerically stable version that subtracts the max
    before exponentiating to prevent overflow.

    Args:
        x: Input array
        axis: Axis along which to compute softmax (default: last axis)

    Returns:
        Softmax probabilities (same shape as input)
    """
    # Subtract max for numerical stability
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def reduce_sum(x: np.ndarray, axis: int = None) -> np.ndarray:
    """
    Sum reduction over specified axis.

    Args:
        x: Input array
        axis: Axis to reduce over. If None, reduces over all elements.

    Returns:
        Sum of elements along specified axis
    """
    return np.sum(x, axis=axis)
