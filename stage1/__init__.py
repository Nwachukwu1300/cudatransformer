# Stage 1: CUDA Kernel Fundamentals
# Provides vector_add, matmul, softmax, and reduce_sum operations

from .kernels import cpu_ops

# Try to import CUDA ops, fall back to CPU-only mode
try:
    from . import cuda_ext as gpu_ops
    CUDA_AVAILABLE = True
except ImportError:
    gpu_ops = None
    CUDA_AVAILABLE = False

__all__ = ['cpu_ops', 'gpu_ops', 'CUDA_AVAILABLE']
