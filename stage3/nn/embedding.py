"""
Embedding layers for transformer.

TokenEmbedding: Maps token IDs to dense vectors.
PositionalEmbedding: Adds position information to token embeddings.
"""

import numpy as np
import sys
import os

# Add stage2 to path for imports
_stage2_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'stage2'))
if _stage2_path not in sys.path:
    sys.path.insert(0, _stage2_path)

from tensor import Tensor

# Import Module using importlib to avoid name collision with local nn package
import importlib.util
_module_spec = importlib.util.spec_from_file_location(
    "stage2_module",
    os.path.join(_stage2_path, 'nn', 'module.py')
)
_module_mod = importlib.util.module_from_spec(_module_spec)
_module_spec.loader.exec_module(_module_mod)
Module = _module_mod.Module


class TokenEmbedding(Module):
    """
    Token embedding layer.

    Maps token indices to dense vectors via embedding lookup.

    Args:
        vocab_size: Size of vocabulary (number of unique tokens)
        embed_dim: Dimension of embedding vectors
    """

    def __init__(self, vocab_size: int, embed_dim: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

        # Initialize embedding matrix with Xavier/Glorot initialization
        scale = np.sqrt(1.0 / embed_dim)
        embedding_data = np.random.randn(vocab_size, embed_dim).astype(np.float32) * scale

        self.embedding = Tensor(embedding_data, requires_grad=True)
        self.register_parameter('embedding', self.embedding)

    def forward(self, token_ids: np.ndarray) -> Tensor:
        """
        Lookup embeddings for token IDs.

        Args:
            token_ids: Integer array of shape (batch_size, seq_len)

        Returns:
            Tensor of shape (batch_size, seq_len, embed_dim)
        """
        # Forward: gather rows from embedding matrix
        output_data = self.embedding.data[token_ids]  # (batch, seq, embed_dim)

        output = Tensor(output_data,
                        requires_grad=self.embedding.requires_grad,
                        _children=(self.embedding,), _op='embedding')

        # Store token_ids for backward
        stored_ids = token_ids.copy()

        def _backward():
            if self.embedding.requires_grad:
                if self.embedding.grad is None:
                    self.embedding.grad = np.zeros_like(self.embedding.data)

                # Scatter gradients back to embedding rows
                # Use np.add.at for accumulation (handles duplicate indices)
                np.add.at(self.embedding.grad, stored_ids, output.grad)

        output._backward = _backward
        return output

    def __repr__(self):
        return f"TokenEmbedding(vocab_size={self.vocab_size}, embed_dim={self.embed_dim})"


class PositionalEmbedding(Module):
    """
    Positional embedding for sequence positions.

    Uses learned position embeddings.

    Args:
        max_seq_len: Maximum sequence length supported
        embed_dim: Dimension of embedding vectors
    """

    def __init__(self, max_seq_len: int, embed_dim: int):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim

        # Learned positional embeddings
        scale = np.sqrt(1.0 / embed_dim)
        pos_data = np.random.randn(max_seq_len, embed_dim).astype(np.float32) * scale

        self.pos_embedding = Tensor(pos_data, requires_grad=True)
        self.register_parameter('pos_embedding', self.pos_embedding)

    def forward(self, seq_len: int) -> Tensor:
        """
        Get positional embeddings for sequence length.

        Args:
            seq_len: Length of sequence

        Returns:
            Tensor of shape (seq_len, embed_dim)
        """
        if seq_len > self.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max {self.max_seq_len}")

        # Slice the position embeddings
        output_data = self.pos_embedding.data[:seq_len]

        output = Tensor(output_data,
                        requires_grad=self.pos_embedding.requires_grad,
                        _children=(self.pos_embedding,), _op='pos_slice')

        # Store seq_len for backward
        stored_seq_len = seq_len

        def _backward():
            if self.pos_embedding.requires_grad:
                if self.pos_embedding.grad is None:
                    self.pos_embedding.grad = np.zeros_like(self.pos_embedding.data)

                # Add gradient to the sliced positions
                self.pos_embedding.grad[:stored_seq_len] += output.grad

        output._backward = _backward
        return output

    def __repr__(self):
        return f"PositionalEmbedding(max_seq_len={self.max_seq_len}, embed_dim={self.embed_dim})"
