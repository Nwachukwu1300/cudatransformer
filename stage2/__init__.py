"""
Stage 2: Deep Learning Engine with Autograd

This module implements a custom autograd engine with:
- Tensor class that tracks operations and enables automatic differentiation
- Neural network layers (Linear, LayerNorm, ReLU, GELU)
- Loss functions (Cross Entropy)
- Optimizers (SGD, Adam)

The module gracefully falls back to CPU implementations when CUDA is not available.
"""

from .kernels import cpu_ops

# Try to import CUDA extensions, fall back to CPU if unavailable
try:
    from . import cuda_ext as gpu_ops
    CUDA_AVAILABLE = True
except ImportError:
    gpu_ops = None
    CUDA_AVAILABLE = False

__all__ = ['cpu_ops', 'gpu_ops', 'CUDA_AVAILABLE']
