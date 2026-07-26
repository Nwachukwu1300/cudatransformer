"""
CPU fallback implementations for Stage 2 operations.

These implementations use NumPy and match the behavior of the CUDA kernels.
They are used when CUDA is not available or for testing/comparison.
"""

import numpy as np


# ================================
# Element-wise Operations
# ================================

def element_wise_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Element-wise multiplication (Hadamard product)."""
    return a * b


def element_wise_divide(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Element-wise division with numerical stability."""
    return a / (b + eps)


def element_wise_subtract(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Element-wise subtraction."""
    return a - b


# ================================
# Scalar Operations
# ================================

def scalar_multiply(a: np.ndarray, scalar: float) -> np.ndarray:
    """Multiply array by scalar."""
    return scalar * a


# ================================
# Mathematical Operations
# ================================

def exp(a: np.ndarray) -> np.ndarray:
    """Element-wise exponential."""
    return np.exp(a)


def log(a: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Element-wise natural logarithm with numerical stability."""
    return np.log(a + eps)


def transpose(a: np.ndarray) -> np.ndarray:
    """Matrix transpose."""
    return a.T


# ================================
# Activation Functions
# ================================

def relu_forward(x: np.ndarray) -> np.ndarray:
    """ReLU activation: max(0, x)."""
    return np.maximum(0, x)


def relu_backward(grad_output: np.ndarray, input: np.ndarray) -> np.ndarray:
    """ReLU backward pass."""
    return grad_output * (input > 0)


def gelu_forward(x: np.ndarray) -> np.ndarray:
    """
    GELU activation.
    GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
    """
    sqrt_2_over_pi = np.sqrt(2.0 / np.pi)
    phi = sqrt_2_over_pi * (x + 0.044715 * x**3)
    return 0.5 * x * (1.0 + np.tanh(phi))


def gelu_backward(grad_output: np.ndarray, input: np.ndarray) -> np.ndarray:
    """
    GELU backward pass.
    """
    sqrt_2_over_pi = np.sqrt(2.0 / np.pi)
    x = input

    # φ(x) = sqrt(2/π) * (x + 0.044715 * x³)
    phi = sqrt_2_over_pi * (x + 0.044715 * x**3)

    # φ'(x) = sqrt(2/π) * (1 + 3 * 0.044715 * x²)
    phi_prime = sqrt_2_over_pi * (1.0 + 3.0 * 0.044715 * x**2)

    # tanh(φ)
    tanh_phi = np.tanh(phi)

    # sech²(φ) = 1 - tanh²(φ)
    sech2_phi = 1.0 - tanh_phi**2

    # GELU'(x) = 0.5 * (1 + tanh(φ)) + 0.5 * x * sech²(φ) * φ'(x)
    gelu_derivative = 0.5 * (1.0 + tanh_phi) + 0.5 * x * sech2_phi * phi_prime

    return grad_output * gelu_derivative


# ================================
# Reduction Operations
# ================================

def reduce_mean(x: np.ndarray, axis: int = -1, keepdims: bool = True) -> np.ndarray:
    """Reduce mean along specified axis."""
    return np.mean(x, axis=axis, keepdims=keepdims)


def reduce_variance(x: np.ndarray, mean: np.ndarray) -> np.ndarray:
    """
    Compute variance given input and pre-computed mean.
    Assumes mean has shape (num_rows,) or (num_rows, 1) for 2D input.
    """
    if mean.ndim == 1:
        mean = mean[:, np.newaxis]
    return np.mean((x - mean)**2, axis=-1, keepdims=True)


def bias_add(input: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """
    Add 1D bias to 2D matrix (broadcast across rows).
    input: (num_rows, num_cols)
    bias: (num_cols,)
    """
    return input + bias


# ================================
# Cross Entropy Loss
# ================================

def cross_entropy_forward(logits: np.ndarray, targets: np.ndarray) -> float:
    """
    Cross entropy loss with numerical stability.
    logits: (num_samples, num_classes)
    targets: (num_samples, num_classes) one-hot encoded
    Returns: scalar loss
    """
    # Numerical stability: subtract max
    max_logits = np.max(logits, axis=-1, keepdims=True)
    shifted_logits = logits - max_logits

    # Log-sum-exp
    log_sum_exp = max_logits + np.log(np.sum(np.exp(shifted_logits), axis=-1, keepdims=True))

    # Log probabilities
    log_probs = shifted_logits - log_sum_exp

    # Cross entropy
    loss = -np.sum(targets * log_probs) / logits.shape[0]

    return float(loss)


def cross_entropy_backward(logits: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """
    Cross entropy backward pass.
    Returns: gradient with respect to logits
    """
    # Compute softmax
    max_logits = np.max(logits, axis=-1, keepdims=True)
    shifted_logits = logits - max_logits
    exp_logits = np.exp(shifted_logits)
    softmax_probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

    # Gradient: (softmax - target) / batch_size
    grad = (softmax_probs - targets) / logits.shape[0]

    return grad


# ================================
# Optimizer Updates
# ================================

def sgd_update(params: np.ndarray, grads: np.ndarray, learning_rate: float) -> np.ndarray:
    """
    SGD parameter update.
    Updates params in-place and returns updated params.
    """
    params -= learning_rate * grads
    return params


def adam_update(params: np.ndarray, grads: np.ndarray,
                m: np.ndarray, v: np.ndarray,
                learning_rate: float, beta1: float, beta2: float,
                eps: float, t: int) -> tuple:
    """
    Adam parameter update.
    Updates params, m, and v in-place and returns them.

    Returns: (updated_params, updated_m, updated_v)
    """
    # Update biased first moment estimate
    m[:] = beta1 * m + (1.0 - beta1) * grads

    # Update biased second raw moment estimate
    v[:] = beta2 * v + (1.0 - beta2) * grads**2

    # Compute bias-corrected first moment estimate
    m_hat = m / (1.0 - beta1**t)

    # Compute bias-corrected second raw moment estimate
    v_hat = v / (1.0 - beta2**t)

    # Update parameters
    params -= learning_rate * m_hat / (np.sqrt(v_hat) + eps)

    return params, m, v
