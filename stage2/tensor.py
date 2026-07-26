"""
Tensor class with automatic differentiation support.

Implements a computation graph with topological sort for backpropagation.
"""

import numpy as np
from typing import Optional, Tuple, Set, Callable, Union
from contextlib import contextmanager


class Tensor:
    """
    Tensor with automatic differentiation.

    Tracks operations in a computation graph and supports backpropagation.
    """

    def __init__(
        self,
        data: Union[np.ndarray, list, float],
        requires_grad: bool = False,
        _children: Tuple['Tensor', ...] = (),
        _op: str = ''
    ):
        """
        Initialize a Tensor.

        Args:
            data: Numerical data (numpy array, list, or scalar)
            requires_grad: Whether to track gradients for this tensor
            _children: Parent tensors in computation graph (internal)
            _op: Operation that created this tensor (internal)
        """
        self.data = np.asarray(data, dtype=np.float32)
        self.grad: Optional[np.ndarray] = None
        self.requires_grad = requires_grad

        # Autograd components
        self._backward: Callable[[], None] = lambda: None
        self._prev: Set[Tensor] = set(_children)
        self._op = _op
        self.is_leaf = len(_children) == 0

    def __repr__(self) -> str:
        return f"Tensor(shape={self.data.shape}, requires_grad={self.requires_grad})"

    @property
    def shape(self) -> Tuple[int, ...]:
        """Return shape of the tensor data."""
        return self.data.shape

    @property
    def ndim(self) -> int:
        """Return number of dimensions."""
        return self.data.ndim

    def reshape(self, *shape) -> 'Tensor':
        """Reshape the tensor."""
        out = Tensor(self.data.reshape(*shape), requires_grad=self.requires_grad,
                     _children=(self,), _op='reshape')

        def _backward():
            if self.requires_grad:
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                self.grad += out.grad.reshape(self.data.shape)

        out._backward = _backward
        return out

    def zero_grad(self):
        """Reset gradients to zero."""
        self.grad = None

    def backward(self, grad: Optional[np.ndarray] = None):
        """
        Compute gradients via backpropagation.

        Args:
            grad: Gradient from upstream (defaults to ones for scalar output)
        """
        # Topological sort
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        # Initialize gradient for output tensor
        if grad is None:
            if self.data.size == 1:
                self.grad = np.ones_like(self.data)
            else:
                raise ValueError("grad must be specified for non-scalar outputs")
        else:
            self.grad = grad

        # Backpropagate
        for node in reversed(topo):
            node._backward()

    # ================================
    # Arithmetic Operations
    # ================================

    def __add__(self, other: Union['Tensor', float]) -> 'Tensor':
        """Addition: self + other"""
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data,
                     requires_grad=self.requires_grad or other.requires_grad,
                     _children=(self, other), _op='+')

        def _backward():
            if self.requires_grad:
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                # Handle broadcasting
                grad_self = out.grad
                if self.data.shape != out.data.shape:
                    # Sum along broadcasted dimensions
                    grad_self = np.sum(grad_self, axis=tuple(range(grad_self.ndim - self.data.ndim)))
                    for i, (dim_self, dim_out) in enumerate(zip(self.data.shape, grad_self.shape)):
                        if dim_self == 1 and dim_out > 1:
                            grad_self = np.sum(grad_self, axis=i, keepdims=True)
                self.grad += grad_self

            if other.requires_grad:
                if other.grad is None:
                    other.grad = np.zeros_like(other.data)
                # Handle broadcasting
                grad_other = out.grad
                if other.data.shape != out.data.shape:
                    grad_other = np.sum(grad_other, axis=tuple(range(grad_other.ndim - other.data.ndim)))
                    for i, (dim_other, dim_out) in enumerate(zip(other.data.shape, grad_other.shape)):
                        if dim_other == 1 and dim_out > 1:
                            grad_other = np.sum(grad_other, axis=i, keepdims=True)
                other.grad += grad_other

        out._backward = _backward
        return out

    def __radd__(self, other: Union['Tensor', float]) -> 'Tensor':
        """Right addition: other + self"""
        return self + other

    def __sub__(self, other: Union['Tensor', float]) -> 'Tensor':
        """Subtraction: self - other"""
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data - other.data,
                     requires_grad=self.requires_grad or other.requires_grad,
                     _children=(self, other), _op='-')

        def _backward():
            if self.requires_grad:
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                self.grad += out.grad

            if other.requires_grad:
                if other.grad is None:
                    other.grad = np.zeros_like(other.data)
                other.grad -= out.grad

        out._backward = _backward
        return out

    def __rsub__(self, other: Union['Tensor', float]) -> 'Tensor':
        """Right subtraction: other - self"""
        other = other if isinstance(other, Tensor) else Tensor(other)
        return other - self

    def __mul__(self, other: Union['Tensor', float]) -> 'Tensor':
        """Element-wise multiplication: self * other"""
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data,
                     requires_grad=self.requires_grad or other.requires_grad,
                     _children=(self, other), _op='*')

        def _backward():
            if self.requires_grad:
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                # d(a*b)/da = b
                grad_self = out.grad * other.data
                if self.data.shape != out.data.shape:
                    grad_self = np.sum(grad_self, axis=tuple(range(grad_self.ndim - self.data.ndim)))
                    for i, (dim_self, dim_out) in enumerate(zip(self.data.shape, grad_self.shape)):
                        if dim_self == 1 and dim_out > 1:
                            grad_self = np.sum(grad_self, axis=i, keepdims=True)
                self.grad += grad_self

            if other.requires_grad:
                if other.grad is None:
                    other.grad = np.zeros_like(other.data)
                # d(a*b)/db = a
                grad_other = out.grad * self.data
                if other.data.shape != out.data.shape:
                    grad_other = np.sum(grad_other, axis=tuple(range(grad_other.ndim - other.data.ndim)))
                    for i, (dim_other, dim_out) in enumerate(zip(other.data.shape, grad_other.shape)):
                        if dim_other == 1 and dim_out > 1:
                            grad_other = np.sum(grad_other, axis=i, keepdims=True)
                other.grad += grad_other

        out._backward = _backward
        return out

    def __rmul__(self, other: Union['Tensor', float]) -> 'Tensor':
        """Right multiplication: other * self"""
        return self * other

    def __truediv__(self, other: Union['Tensor', float]) -> 'Tensor':
        """Element-wise division: self / other"""
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data / (other.data + 1e-8),  # Add epsilon for stability
                     requires_grad=self.requires_grad or other.requires_grad,
                     _children=(self, other), _op='/')

        def _backward():
            if self.requires_grad:
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                # d(a/b)/da = 1/b
                self.grad += out.grad / (other.data + 1e-8)

            if other.requires_grad:
                if other.grad is None:
                    other.grad = np.zeros_like(other.data)
                # d(a/b)/db = -a/b²
                other.grad -= out.grad * self.data / ((other.data + 1e-8) ** 2)

        out._backward = _backward
        return out

    def __matmul__(self, other: 'Tensor') -> 'Tensor':
        """Matrix multiplication: self @ other"""
        out = Tensor(self.data @ other.data,
                     requires_grad=self.requires_grad or other.requires_grad,
                     _children=(self, other), _op='@')

        def _backward():
            if self.requires_grad:
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                # d(A@B)/dA = grad_out @ B.T
                self.grad += out.grad @ other.data.T

            if other.requires_grad:
                if other.grad is None:
                    other.grad = np.zeros_like(other.data)
                # d(A@B)/dB = A.T @ grad_out
                other.grad += self.data.T @ out.grad

        out._backward = _backward
        return out

    def sum(self, axis: Optional[int] = None, keepdims: bool = False) -> 'Tensor':
        """Sum along specified axis."""
        out = Tensor(np.sum(self.data, axis=axis, keepdims=keepdims),
                     requires_grad=self.requires_grad,
                     _children=(self,), _op='sum')

        def _backward():
            if self.requires_grad:
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                # Gradient of sum is broadcast back to original shape
                grad = out.grad
                if not keepdims and axis is not None:
                    grad = np.expand_dims(grad, axis=axis)
                self.grad += np.broadcast_to(grad, self.data.shape)

        out._backward = _backward
        return out

    def mean(self, axis: Optional[int] = None, keepdims: bool = False) -> 'Tensor':
        """Mean along specified axis."""
        out = Tensor(np.mean(self.data, axis=axis, keepdims=keepdims),
                     requires_grad=self.requires_grad,
                     _children=(self,), _op='mean')

        def _backward():
            if self.requires_grad:
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                # Gradient of mean is 1/n broadcast back
                grad = out.grad
                if not keepdims and axis is not None:
                    grad = np.expand_dims(grad, axis=axis)
                n = self.data.shape[axis] if axis is not None else self.data.size
                self.grad += np.broadcast_to(grad, self.data.shape) / n

        out._backward = _backward
        return out

    def sqrt(self) -> 'Tensor':
        """Square root."""
        out = Tensor(np.sqrt(self.data + 1e-8),  # Add epsilon for stability
                     requires_grad=self.requires_grad,
                     _children=(self,), _op='sqrt')

        def _backward():
            if self.requires_grad:
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                # d(sqrt(x))/dx = 1/(2*sqrt(x))
                self.grad += out.grad / (2 * np.sqrt(self.data + 1e-8))

        out._backward = _backward
        return out

    def __pow__(self, power: float) -> 'Tensor':
        """Power operation: self ** power"""
        out = Tensor(self.data ** power,
                     requires_grad=self.requires_grad,
                     _children=(self,), _op=f'**{power}')

        def _backward():
            if self.requires_grad:
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                # d(x^n)/dx = n*x^(n-1)
                self.grad += out.grad * power * (self.data ** (power - 1))

        out._backward = _backward
        return out

    def __neg__(self) -> 'Tensor':
        """Negation: -self"""
        return self * -1


# ================================
# Context Manager for no_grad
# ================================

_grad_enabled = True


@contextmanager
def no_grad():
    """Context manager to disable gradient tracking."""
    global _grad_enabled
    prev = _grad_enabled
    _grad_enabled = False
    try:
        yield
    finally:
        _grad_enabled = prev


def grad_enabled() -> bool:
    """Check if gradient tracking is enabled."""
    return _grad_enabled
