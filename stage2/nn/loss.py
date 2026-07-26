"""
Loss functions.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tensor import Tensor
from nn.module import Module


class CrossEntropyLoss(Module):
    """
    Cross Entropy Loss.

    Combines softmax and negative log likelihood in a numerically stable way.

    Loss = -sum(target * log(softmax(logits))) / batch_size

    Args:
        logits: Unnormalized predictions of shape (batch_size, num_classes)
        targets: Ground truth of shape (batch_size, num_classes) (one-hot encoded)
                or (batch_size,) (class indices)

    Returns:
        Scalar loss value
    """

    def __init__(self):
        super().__init__()

    def forward(self, logits: Tensor, targets: np.ndarray) -> Tensor:
        """
        Forward pass.

        Args:
            logits: Predicted unnormalized logits (batch_size, num_classes)
            targets: Ground truth labels
                    - If 1D (batch_size,): class indices
                    - If 2D (batch_size, num_classes): one-hot encoded

        Returns:
            Scalar loss tensor
        """
        batch_size = logits.shape[0]
        num_classes = logits.shape[1]

        # Convert targets to one-hot if necessary
        if targets.ndim == 1:
            # Convert indices to one-hot
            targets_onehot = np.zeros((batch_size, num_classes), dtype=np.float32)
            targets_onehot[np.arange(batch_size), targets] = 1.0
        else:
            targets_onehot = targets.astype(np.float32)

        # Numerically stable cross entropy using log-sum-exp trick
        # 1. Find max for numerical stability
        max_logits = np.max(logits.data, axis=-1, keepdims=True)

        # 2. Subtract max and compute exp
        shifted_logits = logits.data - max_logits
        exp_logits = np.exp(shifted_logits)

        # 3. Compute softmax
        softmax_probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

        # 4. Compute log probabilities
        log_probs = shifted_logits - np.log(np.sum(exp_logits, axis=-1, keepdims=True))

        # 5. Compute cross entropy loss
        loss_data = -np.sum(targets_onehot * log_probs) / batch_size

        # Create loss tensor
        loss = Tensor(loss_data,
                     requires_grad=logits.requires_grad,
                     _children=(logits,), _op='cross_entropy')

        # Store softmax probs and targets for backward
        softmax = softmax_probs
        targets_stored = targets_onehot

        def _backward():
            if logits.requires_grad:
                if logits.grad is None:
                    logits.grad = np.zeros_like(logits.data)

                # Gradient: (softmax - target) / batch_size
                grad_logits = (softmax - targets_stored) / batch_size

                # Multiply by upstream gradient
                if loss.grad is not None:
                    grad_logits = grad_logits * loss.grad

                logits.grad += grad_logits

        loss._backward = _backward
        return loss

    def __repr__(self):
        return "CrossEntropyLoss()"
