"""
Adam optimizer.
"""

import numpy as np
from typing import List
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tensor import Tensor


class Adam:
    """
    Adam (Adaptive Moment Estimation) optimizer.

    Implements Adam algorithm with bias correction.

    Args:
        params: Iterable of parameters to optimize
        lr: Learning rate (default: 0.001)
        beta1: Coefficient for computing running average of gradient (default: 0.9)
        beta2: Coefficient for computing running average of squared gradient (default: 0.999)
        eps: Term added to denominator for numerical stability (default: 1e-8)
    """

    def __init__(
        self,
        params: List[Tensor],
        lr: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8
    ):
        self.params = list(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        # Initialize state
        self.t = 0  # Timestep
        self.m = [np.zeros_like(param.data) for param in self.params]  # First moment
        self.v = [np.zeros_like(param.data) for param in self.params]  # Second moment

    def step(self):
        """Perform a single optimization step."""
        self.t += 1

        for i, param in enumerate(self.params):
            if param.grad is None:
                continue

            grad = param.grad

            # Update biased first moment estimate
            # m = beta1 * m + (1 - beta1) * grad
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad

            # Update biased second raw moment estimate
            # v = beta2 * v + (1 - beta2) * grad^2
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * grad ** 2

            # Compute bias-corrected first moment estimate
            # m_hat = m / (1 - beta1^t)
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)

            # Compute bias-corrected second raw moment estimate
            # v_hat = v / (1 - beta2^t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            # Update parameters
            # param -= lr * m_hat / (sqrt(v_hat) + eps)
            param.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def zero_grad(self):
        """Zero out gradients for all parameters."""
        for param in self.params:
            param.zero_grad()

    def __repr__(self):
        return (f"Adam(lr={self.lr}, beta1={self.beta1}, "
                f"beta2={self.beta2}, eps={self.eps})")
