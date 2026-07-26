"""
Quick test of Stage 2 components.

Tests basic functionality of Tensor, layers, and autograd.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'stage2'))

from tensor import Tensor
from nn.linear import Linear
from nn.activations import ReLU
from nn.loss import CrossEntropyLoss
from optim.adam import Adam


def test_tensor_autograd():
    """Test basic tensor operations and autograd."""
    print("Testing Tensor autograd...")

    # Test addition
    a = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    b = Tensor([4.0, 5.0, 6.0], requires_grad=True)
    c = a + b
    loss = c.sum()
    loss.backward()

    print(f"  a = {a.data}, grad = {a.grad}")
    print(f"  b = {b.data}, grad = {b.grad}")
    assert np.allclose(a.grad, [1.0, 1.0, 1.0]), "Addition gradient failed"
    assert np.allclose(b.grad, [1.0, 1.0, 1.0]), "Addition gradient failed"
    print("  ✓ Addition and sum gradient correct")

    # Test multiplication
    a = Tensor([2.0, 3.0], requires_grad=True)
    b = Tensor([4.0, 5.0], requires_grad=True)
    c = a * b
    loss = c.sum()
    loss.backward()

    assert np.allclose(a.grad, b.data), "Multiplication gradient failed"
    assert np.allclose(b.grad, a.data), "Multiplication gradient failed"
    print("  ✓ Multiplication gradient correct")

    # Test matrix multiplication
    a = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    b = Tensor([[5.0, 6.0], [7.0, 8.0]], requires_grad=True)
    c = a @ b
    loss = c.sum()
    loss.backward()

    expected_grad_a = np.array([[11.0, 15.0], [11.0, 15.0]])
    expected_grad_b = np.array([[4.0, 4.0], [6.0, 6.0]])
    assert np.allclose(a.grad, expected_grad_a), "Matmul gradient failed"
    assert np.allclose(b.grad, expected_grad_b), "Matmul gradient failed"
    print("  ✓ Matrix multiplication gradient correct")

    print("  All tensor autograd tests passed!\n")


def test_linear_layer():
    """Test Linear layer."""
    print("Testing Linear layer...")

    # Create layer
    layer = Linear(3, 2)

    # Forward pass
    x = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=False)
    y = layer(x)

    assert y.shape == (2, 2), f"Output shape incorrect: {y.shape}"
    print(f"  Input shape: {x.shape}, Output shape: {y.shape}")
    print("  ✓ Linear layer forward pass works")

    # Backward pass
    loss = y.sum()
    loss.backward()

    assert layer.weight.grad is not None, "Weight gradient not computed"
    assert layer.bias.grad is not None, "Bias gradient not computed"
    print("  ✓ Linear layer backward pass works")
    print(f"  Weight grad shape: {layer.weight.grad.shape}")
    print(f"  Bias grad shape: {layer.bias.grad.shape}\n")


def test_relu():
    """Test ReLU activation."""
    print("Testing ReLU activation...")

    relu = ReLU()
    x = Tensor([[-1.0, 2.0, -3.0, 4.0]], requires_grad=True)
    y = relu(x)

    expected_output = np.array([[0.0, 2.0, 0.0, 4.0]])
    assert np.allclose(y.data, expected_output), "ReLU forward pass failed"
    print(f"  Input: {x.data}")
    print(f"  Output: {y.data}")
    print("  ✓ ReLU forward pass correct")

    # Backward
    loss = y.sum()
    loss.backward()

    expected_grad = np.array([[0.0, 1.0, 0.0, 1.0]])
    assert np.allclose(x.grad, expected_grad), "ReLU backward pass failed"
    print(f"  Gradient: {x.grad}")
    print("  ✓ ReLU backward pass correct\n")


def test_cross_entropy():
    """Test Cross Entropy loss."""
    print("Testing Cross Entropy loss...")

    criterion = CrossEntropyLoss()

    # Create dummy logits and targets
    logits = Tensor([[2.0, 1.0, 0.1], [0.1, 2.0, 1.0]], requires_grad=True)
    targets = np.array([0, 1])  # Class indices

    # Forward
    loss = criterion(logits, targets)
    print(f"  Loss: {loss.data:.4f}")
    print("  ✓ Cross entropy forward pass works")

    # Backward
    loss.backward()
    assert logits.grad is not None, "Logits gradient not computed"
    print(f"  Logits gradient:\n{logits.grad}")
    print("  ✓ Cross entropy backward pass works\n")


def test_simple_training():
    """Test simple training loop on toy data."""
    print("Testing simple training loop...")

    # Create simple model
    model_layers = [
        Linear(2, 4),
        ReLU(),
        Linear(4, 2)
    ]

    # Create toy data (XOR problem)
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    y = np.array([0, 1, 1, 0])  # XOR labels

    # Get parameters from all layers
    params = []
    for layer in model_layers:
        if hasattr(layer, 'parameters'):
            params.extend(list(layer.parameters()))

    optimizer = Adam(params, lr=0.1)
    criterion = CrossEntropyLoss()

    # Train for a few iterations
    print("  Training on XOR problem for 10 iterations...")
    for iteration in range(10):
        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        x = Tensor(X, requires_grad=False)
        for layer in model_layers:
            x = layer(x)
        logits = x

        # Compute loss
        loss = criterion(logits, y)

        # Backward
        loss.backward()

        # Update
        optimizer.step()

        if (iteration + 1) % 5 == 0:
            print(f"    Iteration {iteration + 1}, Loss: {loss.data:.4f}")

    print("  ✓ Simple training loop works\n")


def main():
    """Run all tests."""
    print("=" * 70)
    print("Stage 2 Component Tests")
    print("=" * 70)
    print()

    test_tensor_autograd()
    test_linear_layer()
    test_relu()
    test_cross_entropy()
    test_simple_training()

    print("=" * 70)
    print("All tests passed! Stage 2 components are working correctly.")
    print("=" * 70)


if __name__ == "__main__":
    main()
