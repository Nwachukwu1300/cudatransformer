"""
Activation functions: ReLU and GELU.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tensor import Tensor
from nn.module import Module


class ReLU(Module):
    """
    Rectified Linear Unit activation function.

    ReLU(x) = max(0, x)
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor

        Returns:
            Output tensor with ReLU applied element-wise
        """
        # ReLU forward
        output_data = np.maximum(0, x.data)
        output = Tensor(output_data,
                       requires_grad=x.requires_grad,
                       _children=(x,), _op='relu')

        def _backward():
            if x.requires_grad:
                if x.grad is None:
                    x.grad = np.zeros_like(x.data)
                # ReLU gradient: pass through where x > 0, zero elsewhere
                x.grad += output.grad * (x.data > 0)

        output._backward = _backward
        return output

    def __repr__(self):
        return "ReLU()"


class GELU(Module):
    """
    Gaussian Error Linear Unit activation function.

    GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
    """

    def __init__(self):
        super().__init__()
        self.sqrt_2_over_pi = np.sqrt(2.0 / np.pi)
        self.coeff = 0.044715

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor

        Returns:
            Output tensor with GELU applied element-wise
        """
        # GELU forward
        x_data = x.data
        phi = self.sqrt_2_over_pi * (x_data + self.coeff * x_data ** 3)
        output_data = 0.5 * x_data * (1.0 + np.tanh(phi))

        output = Tensor(output_data,
                       requires_grad=x.requires_grad,
                       _children=(x,), _op='gelu')

        def _backward():
            if x.requires_grad:
                if x.grad is None:
                    x.grad = np.zeros_like(x.data)

                # GELU gradient
                x_data = x.data

                # φ(x) = sqrt(2/π) * (x + 0.044715 * x³)
                phi = self.sqrt_2_over_pi * (x_data + self.coeff * x_data ** 3)

                # φ'(x) = sqrt(2/π) * (1 + 3 * 0.044715 * x²)
                phi_prime = self.sqrt_2_over_pi * (1.0 + 3.0 * self.coeff * x_data ** 2)

                # tanh(φ)
                tanh_phi = np.tanh(phi)

                # sech²(φ) = 1 - tanh²(φ)
                sech2_phi = 1.0 - tanh_phi ** 2

                # GELU'(x) = 0.5 * (1 + tanh(φ)) + 0.5 * x * sech²(φ) * φ'(x)
                gelu_derivative = 0.5 * (1.0 + tanh_phi) + \
                                 0.5 * x_data * sech2_phi * phi_prime

                x.grad += output.grad * gelu_derivative

        output._backward = _backward
        return output

    def __repr__(self):
        return "GELU()"
