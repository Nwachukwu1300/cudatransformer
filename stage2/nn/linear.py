"""
Linear (fully connected) layer.

Implements: output = input @ weight.T + bias
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tensor import Tensor
from nn.module import Module


class Linear(Module):
    """
    Linear (fully connected) layer.

    Applies a linear transformation: y = xW^T + b

    Args:
        in_features: Size of each input sample
        out_features: Size of each output sample
        bias: If True, adds a learnable bias to the output

    Attributes:
        weight: Learnable weights of shape (out_features, in_features)
        bias: Learnable bias of shape (out_features,)
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Initialize weights using He initialization
        # He init: weight ~ N(0, sqrt(2 / in_features))
        std = np.sqrt(2.0 / in_features)
        weight_data = np.random.randn(out_features, in_features).astype(np.float32) * std

        self.weight = Tensor(weight_data, requires_grad=True)
        self.register_parameter('weight', self.weight)

        if bias:
            bias_data = np.zeros(out_features, dtype=np.float32)
            self.bias = Tensor(bias_data, requires_grad=True)
            self.register_parameter('bias', self.bias)
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, in_features)

        Returns:
            Output tensor of shape (batch_size, out_features)
        """
        # Ensure input is 2D
        if x.ndim == 1:
            x = x.reshape(1, -1)

        # Matrix multiplication: x @ weight.T
        # x: (batch, in_features)
        # weight: (out_features, in_features)
        # weight.T: (in_features, out_features)
        # output: (batch, out_features)
        output = x @ Tensor(self.weight.data.T, requires_grad=self.weight.requires_grad)

        # Need to manually set up backward for weight
        # Since we used weight.T, we need to fix the gradient computation
        out_temp = output

        # Properly compute gradients for weight
        output = Tensor(out_temp.data,
                       requires_grad=x.requires_grad or self.weight.requires_grad,
                       _children=(x, self.weight), _op='linear')

        def _backward():
            if x.requires_grad:
                if x.grad is None:
                    x.grad = np.zeros_like(x.data)
                # dL/dx = dL/dout @ weight
                x.grad += output.grad @ self.weight.data

            if self.weight.requires_grad:
                if self.weight.grad is None:
                    self.weight.grad = np.zeros_like(self.weight.data)
                # dL/dW = dL/dout.T @ x
                self.weight.grad += output.grad.T @ x.data

        output._backward = _backward

        # Add bias if present
        if self.bias is not None:
            # Broadcast bias across batch dimension
            output = output + self.bias

        return output

    def __repr__(self):
        return f"Linear(in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None})"
