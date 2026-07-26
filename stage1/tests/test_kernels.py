"""
Correctness tests for Stage 1 kernels.

Tests CPU implementations against NumPy reference implementations.
When CUDA is available, also tests GPU implementations against CPU.
"""

import numpy as np
import pytest
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from stage1.kernels import cpu_ops

# Try to import CUDA extension
try:
    from stage1 import cuda_ext as gpu_ops
    CUDA_AVAILABLE = True
except ImportError:
    gpu_ops = None
    CUDA_AVAILABLE = False


class TestVectorAdd:
    """Tests for vector addition."""

    def test_basic(self):
        """Basic vector addition."""
        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        b = np.array([4.0, 5.0, 6.0], dtype=np.float32)
        expected = np.array([5.0, 7.0, 9.0], dtype=np.float32)

        result = cpu_ops.vector_add(a, b)
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_random_small(self):
        """Random small vectors."""
        np.random.seed(42)
        a = np.random.randn(100).astype(np.float32)
        b = np.random.randn(100).astype(np.float32)

        result = cpu_ops.vector_add(a, b)
        expected = a + b
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_random_large(self):
        """Random large vectors."""
        np.random.seed(42)
        a = np.random.randn(100000).astype(np.float32)
        b = np.random.randn(100000).astype(np.float32)

        result = cpu_ops.vector_add(a, b)
        expected = a + b
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_zeros(self):
        """Zero vectors."""
        a = np.zeros(50, dtype=np.float32)
        b = np.zeros(50, dtype=np.float32)

        result = cpu_ops.vector_add(a, b)
        np.testing.assert_allclose(result, np.zeros(50), rtol=1e-5)

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_gpu_matches_cpu(self):
        """GPU result matches CPU result."""
        np.random.seed(42)
        a = np.random.randn(10000).astype(np.float32)
        b = np.random.randn(10000).astype(np.float32)

        cpu_result = cpu_ops.vector_add(a, b)
        gpu_result = gpu_ops.vector_add(a, b)
        np.testing.assert_allclose(gpu_result, cpu_result, rtol=1e-5)


class TestMatmul:
    """Tests for matrix multiplication."""

    def test_basic(self):
        """Basic 2x2 multiplication."""
        A = np.array([[1, 2], [3, 4]], dtype=np.float32)
        B = np.array([[5, 6], [7, 8]], dtype=np.float32)
        expected = np.array([[19, 22], [43, 50]], dtype=np.float32)

        result = cpu_ops.matmul(A, B)
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_identity(self):
        """Multiplication by identity matrix."""
        A = np.random.randn(10, 10).astype(np.float32)
        I = np.eye(10, dtype=np.float32)

        result = cpu_ops.matmul(A, I)
        np.testing.assert_allclose(result, A, rtol=1e-5)

    def test_random_square(self):
        """Random square matrices."""
        np.random.seed(42)
        A = np.random.randn(64, 64).astype(np.float32)
        B = np.random.randn(64, 64).astype(np.float32)

        result = cpu_ops.matmul(A, B)
        expected = np.matmul(A, B)
        np.testing.assert_allclose(result, expected, rtol=1e-4)

    def test_random_rectangular(self):
        """Random rectangular matrices."""
        np.random.seed(42)
        A = np.random.randn(32, 64).astype(np.float32)
        B = np.random.randn(64, 48).astype(np.float32)

        result = cpu_ops.matmul(A, B)
        expected = np.matmul(A, B)
        np.testing.assert_allclose(result, expected, rtol=1e-4)

    def test_large_matrices(self):
        """Large matrix multiplication."""
        np.random.seed(42)
        A = np.random.randn(256, 256).astype(np.float32)
        B = np.random.randn(256, 256).astype(np.float32)

        result = cpu_ops.matmul(A, B)
        expected = np.matmul(A, B)
        np.testing.assert_allclose(result, expected, rtol=1e-4)

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_gpu_matches_cpu(self):
        """GPU result matches CPU result."""
        np.random.seed(42)
        A = np.random.randn(128, 128).astype(np.float32)
        B = np.random.randn(128, 128).astype(np.float32)

        cpu_result = cpu_ops.matmul(A, B)
        gpu_result = gpu_ops.matmul(A, B)
        np.testing.assert_allclose(gpu_result, cpu_result, rtol=1e-4)


class TestSoftmax:
    """Tests for softmax."""

    def test_basic(self):
        """Basic softmax."""
        x = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        result = cpu_ops.softmax(x, axis=-1)

        # Sum should be 1
        np.testing.assert_allclose(result.sum(axis=-1), 1.0, rtol=1e-5)

        # Largest input should have largest probability
        assert result[0, 2] > result[0, 1] > result[0, 0]

    def test_matches_scipy(self):
        """Matches reference implementation."""
        np.random.seed(42)
        x = np.random.randn(10, 100).astype(np.float32)

        result = cpu_ops.softmax(x, axis=-1)

        # Manual reference implementation
        x_max = x.max(axis=-1, keepdims=True)
        exp_x = np.exp(x - x_max)
        expected = exp_x / exp_x.sum(axis=-1, keepdims=True)

        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_numerical_stability(self):
        """Large values don't cause overflow."""
        x = np.array([[1000.0, 1001.0, 1002.0]], dtype=np.float32)
        result = cpu_ops.softmax(x, axis=-1)

        # Should not be NaN or Inf
        assert np.all(np.isfinite(result))
        np.testing.assert_allclose(result.sum(axis=-1), 1.0, rtol=1e-5)

    def test_uniform_input(self):
        """Uniform input gives uniform output."""
        x = np.ones((5, 10), dtype=np.float32) * 3.0
        result = cpu_ops.softmax(x, axis=-1)

        expected = np.ones((5, 10), dtype=np.float32) / 10.0
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_batch(self):
        """Multiple rows processed correctly."""
        np.random.seed(42)
        x = np.random.randn(100, 50).astype(np.float32)

        result = cpu_ops.softmax(x, axis=-1)

        # Each row should sum to 1
        np.testing.assert_allclose(result.sum(axis=-1), np.ones(100), rtol=1e-5)

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_gpu_matches_cpu(self):
        """GPU result matches CPU result."""
        np.random.seed(42)
        x = np.random.randn(64, 128).astype(np.float32)

        cpu_result = cpu_ops.softmax(x, axis=-1)
        gpu_result = gpu_ops.softmax(x)
        np.testing.assert_allclose(gpu_result, cpu_result, rtol=1e-4)


class TestReduceSum:
    """Tests for sum reduction."""

    def test_basic(self):
        """Basic sum."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        result = cpu_ops.reduce_sum(x)
        assert result == 15.0

    def test_single_element(self):
        """Single element array."""
        x = np.array([42.0], dtype=np.float32)
        result = cpu_ops.reduce_sum(x)
        assert result == 42.0

    def test_random(self):
        """Random array."""
        np.random.seed(42)
        x = np.random.randn(1000).astype(np.float32)

        result = cpu_ops.reduce_sum(x)
        expected = np.sum(x)
        np.testing.assert_allclose(result, expected, rtol=1e-4)

    def test_large_array(self):
        """Large array."""
        np.random.seed(42)
        x = np.random.randn(1000000).astype(np.float32)

        result = cpu_ops.reduce_sum(x)
        expected = np.sum(x)
        # Larger tolerance due to floating point accumulation
        np.testing.assert_allclose(result, expected, rtol=1e-3)

    def test_zeros(self):
        """Zero array."""
        x = np.zeros(100, dtype=np.float32)
        result = cpu_ops.reduce_sum(x)
        assert result == 0.0

    def test_all_ones(self):
        """All ones."""
        x = np.ones(1000, dtype=np.float32)
        result = cpu_ops.reduce_sum(x)
        np.testing.assert_allclose(result, 1000.0, rtol=1e-5)

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_gpu_matches_cpu(self):
        """GPU result matches CPU result."""
        np.random.seed(42)
        x = np.random.randn(100000).astype(np.float32)

        cpu_result = cpu_ops.reduce_sum(x)
        gpu_result = gpu_ops.reduce_sum(x)
        np.testing.assert_allclose(gpu_result, cpu_result, rtol=1e-3)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
