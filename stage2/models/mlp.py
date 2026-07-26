"""
Multi-Layer Perceptron (MLP) for MNIST classification.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tensor import Tensor
from nn.module import Module
from nn.linear import Linear
from nn.activations import ReLU


class MLP(Module):
    """
    Simple MLP for MNIST classification.

    Architecture:
        Input (784) -> Linear (256) -> ReLU -> Linear (128) -> ReLU -> Linear (10)

    Args:
        input_size: Size of input features (default: 784 for MNIST)
        hidden1_size: Size of first hidden layer (default: 256)
        hidden2_size: Size of second hidden layer (default: 128)
        num_classes: Number of output classes (default: 10 for MNIST)
    """

    def __init__(
        self,
        input_size: int = 784,
        hidden1_size: int = 256,
        hidden2_size: int = 128,
        num_classes: int = 10
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden1_size = hidden1_size
        self.hidden2_size = hidden2_size
        self.num_classes = num_classes

        # Layers
        self.fc1 = Linear(input_size, hidden1_size)
        self.relu1 = ReLU()
        self.fc2 = Linear(hidden1_size, hidden2_size)
        self.relu2 = ReLU()
        self.fc3 = Linear(hidden2_size, num_classes)

        # Register as submodules
        self.register_module('fc1', self.fc1)
        self.register_module('relu1', self.relu1)
        self.register_module('fc2', self.fc2)
        self.register_module('relu2', self.relu2)
        self.register_module('fc3', self.fc3)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, input_size) or (batch_size, 28, 28)

        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        # Flatten input if necessary (batch_size, 28, 28) -> (batch_size, 784)
        if x.ndim == 3:
            batch_size = x.shape[0]
            x = x.reshape(batch_size, -1)
        elif x.ndim == 2 and x.shape[1] != self.input_size:
            # If 2D but wrong size, try to reshape
            batch_size = x.shape[0]
            x = x.reshape(batch_size, -1)

        # Forward pass through layers
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)

        return x

    def count_parameters(self) -> int:
        """Count total number of trainable parameters."""
        total = 0
        for param in self.parameters():
            total += param.data.size
        return total

    def __repr__(self):
        param_count = self.count_parameters()
        return (f"MLP(\n"
                f"  Input: {self.input_size}\n"
                f"  Hidden 1: {self.hidden1_size} + ReLU\n"
                f"  Hidden 2: {self.hidden2_size} + ReLU\n"
                f"  Output: {self.num_classes}\n"
                f"  Total Parameters: {param_count:,}\n"
                f")")
