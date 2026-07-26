"""
Multi-Head Attention for transformer.

Implements scaled dot-product attention with causal masking.
"""

import numpy as np
import sys
import os
from typing import Optional
import importlib.util

# Add stage2 to path for imports
_stage2_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'stage2'))
if _stage2_path not in sys.path:
    sys.path.insert(0, _stage2_path)

from tensor import Tensor

# Import Module and Linear using importlib to avoid name collision with local nn package
def _load_stage2_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_stage2_path, 'nn', filename))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_module_mod = _load_stage2_module("stage2_module", "module.py")
_linear_mod = _load_stage2_module("stage2_linear", "linear.py")

Module = _module_mod.Module
Linear = _linear_mod.Linear


def create_causal_mask(seq_len: int) -> np.ndarray:
    """
    Create causal (autoregressive) attention mask.

    For position i, can only attend to positions 0, 1, ..., i.
    Mask is additive: add -1e9 to positions that should not be attended to.

    Args:
        seq_len: Sequence length

    Returns:
        Mask array of shape (seq_len, seq_len)
    """
    mask = np.triu(np.ones((seq_len, seq_len)), k=1) * (-1e9)
    return mask.astype(np.float32)


def scaled_dot_product_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    mask: Optional[np.ndarray] = None
) -> Tensor:
    """
    Compute scaled dot-product attention.

    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V

    Args:
        query: Query tensor of shape (batch, heads, seq_len, head_dim)
        key: Key tensor of shape (batch, heads, seq_len, head_dim)
        value: Value tensor of shape (batch, heads, seq_len, head_dim)
        mask: Optional attention mask of shape (seq_len, seq_len)

    Returns:
        Attention output tensor of shape (batch, heads, seq_len, head_dim)
    """
    batch_size, num_heads, seq_len, head_dim = query.shape
    scale = 1.0 / np.sqrt(head_dim)

    # Forward pass
    # scores: (batch, heads, seq_q, seq_k)
    # Q @ K^T where K^T swaps last two dims
    scores_data = (query.data @ np.swapaxes(key.data, -2, -1)) * scale

    # Apply mask if provided
    if mask is not None:
        scores_data = scores_data + mask

    # Softmax over last dimension (keys)
    # Numerically stable softmax using max subtraction
    max_scores = np.max(scores_data, axis=-1, keepdims=True)
    exp_scores = np.exp(scores_data - max_scores)
    probs_data = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

    # Weighted sum of values
    output_data = probs_data @ value.data

    output = Tensor(output_data,
                    requires_grad=query.requires_grad or key.requires_grad or value.requires_grad,
                    _children=(query, key, value), _op='attention')

    # Store intermediates for backward
    stored_probs = probs_data
    stored_scale = scale

    def _backward():
        grad_output = output.grad

        # Gradient w.r.t. value: probs^T @ grad_output
        # probs: (batch, heads, seq_q, seq_k)
        # grad_output: (batch, heads, seq_q, head_dim)
        # V grad: (batch, heads, seq_k, head_dim)
        if value.requires_grad:
            if value.grad is None:
                value.grad = np.zeros_like(value.data)
            # Transpose probs to (batch, heads, seq_k, seq_q) then @ grad_output
            value.grad += np.swapaxes(stored_probs, -2, -1) @ grad_output

        # Gradient w.r.t. probs: grad_output @ V^T
        # grad_probs: (batch, heads, seq_q, seq_k)
        grad_probs = grad_output @ np.swapaxes(value.data, -2, -1)

        # Backward through softmax
        # For softmax: d_scores = probs * (d_probs - sum(d_probs * probs, axis=-1, keepdims=True))
        sum_grad_probs_probs = np.sum(grad_probs * stored_probs, axis=-1, keepdims=True)
        grad_scores = stored_probs * (grad_probs - sum_grad_probs_probs)

        # Scale gradient
        grad_scores = grad_scores * stored_scale

        # Gradient w.r.t. query: grad_scores @ K
        if query.requires_grad:
            if query.grad is None:
                query.grad = np.zeros_like(query.data)
            query.grad += grad_scores @ key.data

        # Gradient w.r.t. key: Q^T @ grad_scores -> need to transpose result
        # grad_scores: (batch, heads, seq_q, seq_k)
        # query: (batch, heads, seq_q, head_dim)
        # We want K grad: (batch, heads, seq_k, head_dim)
        # Q^T @ grad_scores gives (batch, heads, head_dim, seq_k), need to transpose
        if key.requires_grad:
            if key.grad is None:
                key.grad = np.zeros_like(key.data)
            # (batch, heads, seq_q, head_dim)^T @ (batch, heads, seq_q, seq_k)
            # = (batch, heads, head_dim, seq_q) @ (batch, heads, seq_q, seq_k)
            # = (batch, heads, head_dim, seq_k)
            # Then transpose to (batch, heads, seq_k, head_dim)
            key.grad += np.swapaxes(np.swapaxes(query.data, -2, -1) @ grad_scores, -2, -1)

    output._backward = _backward
    return output


class MultiHeadAttention(Module):
    """
    Multi-Head Attention.

    Splits input into multiple heads, applies attention, concatenates results.

    Args:
        embed_dim: Total embedding dimension
        num_heads: Number of attention heads
    """

    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Q, K, V projections
        self.W_q = Linear(embed_dim, embed_dim, bias=False)
        self.W_k = Linear(embed_dim, embed_dim, bias=False)
        self.W_v = Linear(embed_dim, embed_dim, bias=False)

        # Output projection
        self.W_o = Linear(embed_dim, embed_dim, bias=False)

        # Register submodules
        self.register_module('W_q', self.W_q)
        self.register_module('W_k', self.W_k)
        self.register_module('W_v', self.W_v)
        self.register_module('W_o', self.W_o)

    def _reshape_for_attention(self, x: Tensor, batch_size: int, seq_len: int) -> Tensor:
        """
        Reshape from (batch * seq, embed) to (batch, heads, seq, head_dim).

        Args:
            x: Input tensor of shape (batch * seq, embed_dim)
            batch_size: Batch size
            seq_len: Sequence length

        Returns:
            Reshaped tensor of shape (batch, heads, seq, head_dim)
        """
        # First reshape to (batch, seq, heads, head_dim)
        x = x.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        # Then transpose to (batch, heads, seq, head_dim)
        x = x.transpose(1, 2)
        return x

    def _reshape_from_attention(self, x: Tensor, batch_size: int, seq_len: int) -> Tensor:
        """
        Reshape from (batch, heads, seq, head_dim) to (batch * seq, embed).

        Args:
            x: Input tensor of shape (batch, heads, seq, head_dim)
            batch_size: Batch size
            seq_len: Sequence length

        Returns:
            Reshaped tensor of shape (batch * seq, embed_dim)
        """
        # Transpose back to (batch, seq, heads, head_dim)
        x = x.transpose(1, 2)
        # Reshape to (batch * seq, embed_dim)
        x = x.reshape(batch_size * seq_len, self.embed_dim)
        return x

    def forward(self, x: Tensor, mask: Optional[np.ndarray] = None) -> Tensor:
        """
        Forward pass for self-attention.

        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim)
            mask: Optional causal mask of shape (seq_len, seq_len)

        Returns:
            Output tensor of shape (batch_size, seq_len, embed_dim)
        """
        batch_size, seq_len, _ = x.shape

        # Reshape input to 2D for Linear layers: (batch * seq, embed_dim)
        x_2d = x.reshape(batch_size * seq_len, self.embed_dim)

        # 1. Linear projections
        Q = self.W_q(x_2d)  # (batch * seq, embed_dim)
        K = self.W_k(x_2d)
        V = self.W_v(x_2d)

        # 2. Reshape for multi-head attention
        Q = self._reshape_for_attention(Q, batch_size, seq_len)
        K = self._reshape_for_attention(K, batch_size, seq_len)
        V = self._reshape_for_attention(V, batch_size, seq_len)

        # 3. Scaled dot-product attention
        attn_output = scaled_dot_product_attention(Q, K, V, mask)

        # 4. Reshape back and apply output projection
        attn_output_2d = self._reshape_from_attention(attn_output, batch_size, seq_len)
        output_2d = self.W_o(attn_output_2d)

        # 5. Reshape back to 3D
        output = output_2d.reshape(batch_size, seq_len, self.embed_dim)

        return output

    def __repr__(self):
        return f"MultiHeadAttention(embed_dim={self.embed_dim}, num_heads={self.num_heads})"
