"""
Stochastic Gradient Descent optimizer.
"""

import numpy as np
from typing import List
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tensor import Tensor


class SGD:
    """
    Stochastic Gradient Descent optimizer.

    Updates parameters using:
    param -= learning_rate * grad

    Args:
        params: Iterable of parameters to optimize
        lr: Learning rate
    """

    def __init__(self, params: List[Tensor], lr: float = 0.01):
        self.params = list(params)
        self.lr = lr

    def step(self):
        """Perform a single optimization step."""
        for param in self.params:
            if param.grad is not None:
                # Update: param -= lr * grad
                param.data -= self.lr * param.grad

    def zero_grad(self):
        """Zero out gradients for all parameters."""
        for param in self.params:
            param.zero_grad()

    def __repr__(self):
        return f"SGD(lr={self.lr})"
