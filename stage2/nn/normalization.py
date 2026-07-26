"""
Layer Normalization.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tensor import Tensor
from nn.module import Module


class LayerNorm(Module):
    """
    Layer Normalization.

    Normalizes inputs across the feature dimension:
    y = gamma * (x - mean) / sqrt(var + eps) + beta

    Args:
        normalized_shape: Size of the feature dimension to normalize over
        eps: Small value added to variance for numerical stability

    Attributes:
        gamma: Learnable scale parameter of shape (normalized_shape,)
        beta: Learnable shift parameter of shape (normalized_shape,)
    """

    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps

        # Initialize gamma to ones and beta to zeros
        gamma_data = np.ones(normalized_shape, dtype=np.float32)
        beta_data = np.zeros(normalized_shape, dtype=np.float32)

        self.gamma = Tensor(gamma_data, requires_grad=True)
        self.beta = Tensor(beta_data, requires_grad=True)

        self.register_parameter('gamma', self.gamma)
        self.register_parameter('beta', self.beta)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (..., normalized_shape)

        Returns:
            Normalized tensor of same shape as input
        """
        # Ensure input has correct feature dimension
        if x.shape[-1] != self.normalized_shape:
            raise ValueError(f"Expected last dimension to be {self.normalized_shape}, "
                           f"got {x.shape[-1]}")

        # Compute mean and variance along last axis
        mean_data = np.mean(x.data, axis=-1, keepdims=True)
        var_data = np.var(x.data, axis=-1, keepdims=True)

        # Normalize
        x_normalized_data = (x.data - mean_data) / np.sqrt(var_data + self.eps)

        # Scale and shift
        output_data = self.gamma.data * x_normalized_data + self.beta.data

        # Create output tensor
        output = Tensor(output_data,
                       requires_grad=x.requires_grad or self.gamma.requires_grad or self.beta.requires_grad,
                       _children=(x, self.gamma, self.beta), _op='layernorm')

        # Store intermediate values for backward pass
        mean = mean_data
        var = var_data
        x_normalized = x_normalized_data
        N = x.shape[-1]  # Feature dimension

        def _backward():
            grad_output = output.grad

            # Gradient w.r.t. gamma
            if self.gamma.requires_grad:
                if self.gamma.grad is None:
                    self.gamma.grad = np.zeros_like(self.gamma.data)
                # dL/dgamma = sum(dL/doutput * x_normalized)
                self.gamma.grad += np.sum(grad_output * x_normalized, axis=tuple(range(grad_output.ndim - 1)))

            # Gradient w.r.t. beta
            if self.beta.requires_grad:
                if self.beta.grad is None:
                    self.beta.grad = np.zeros_like(self.beta.data)
                # dL/dbeta = sum(dL/doutput)
                self.beta.grad += np.sum(grad_output, axis=tuple(range(grad_output.ndim - 1)))

            # Gradient w.r.t. input
            if x.requires_grad:
                if x.grad is None:
                    x.grad = np.zeros_like(x.data)

                # dL/d(x_normalized)
                d_xnorm = grad_output * self.gamma.data

                # Gradient through normalization
                # Using the efficient formulation:
                # dL/dx = (1/N) * (1/sqrt(var+eps)) * (N*d_xnorm - sum(d_xnorm) - x_norm*sum(d_xnorm*x_norm))

                std_inv = 1.0 / np.sqrt(var + self.eps)

                # Compute terms
                sum_d_xnorm = np.sum(d_xnorm, axis=-1, keepdims=True)
                sum_d_xnorm_x_xnorm = np.sum(d_xnorm * x_normalized, axis=-1, keepdims=True)

                # Gradient w.r.t. input
                dx = (1.0 / N) * std_inv * (
                    N * d_xnorm - sum_d_xnorm - x_normalized * sum_d_xnorm_x_xnorm
                )

                x.grad += dx

        output._backward = _backward
        return output

    def __repr__(self):
        return f"LayerNorm(normalized_shape={self.normalized_shape}, eps={self.eps})"
