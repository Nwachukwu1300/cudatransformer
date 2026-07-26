"""
Base Module class for neural network layers.

Provides parameter management and forward pass interface.
"""

from typing import List, Iterator
import sys
import os

# Add parent directory to path to import Tensor
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tensor import Tensor


class Module:
    """
    Base class for all neural network modules.

    Subclasses should override the forward() method.
    """

    def __init__(self):
        self._parameters = []
        self._modules = []

    def parameters(self) -> Iterator[Tensor]:
        """
        Return an iterator over module parameters.

        Recursively yields parameters from this module and all submodules.
        """
        # Yield parameters from this module
        for param in self._parameters:
            yield param

        # Recursively yield parameters from submodules
        # Use hasattr check instead of isinstance for compatibility with importlib
        for module in self._modules:
            if hasattr(module, 'parameters'):
                yield from module.parameters()

    def register_parameter(self, name: str, param: Tensor):
        """Register a parameter with this module."""
        setattr(self, name, param)
        self._parameters.append(param)

    def register_module(self, name: str, module: 'Module'):
        """Register a submodule with this module."""
        setattr(self, name, module)
        self._modules.append(module)

    def zero_grad(self):
        """Zero out gradients for all parameters."""
        for param in self.parameters():
            param.zero_grad()

    def forward(self, *args, **kwargs):
        """
        Forward pass.

        Should be overridden by subclasses.
        """
        raise NotImplementedError("Subclasses must implement forward()")

    def __call__(self, *args, **kwargs):
        """Allow module to be called like a function."""
        return self.forward(*args, **kwargs)

    def __repr__(self):
        return f"{self.__class__.__name__}()"
