"""
Transformer components: FeedForward, TransformerBlock, TransformerDecoder.
"""

import numpy as np
import sys
import os
from typing import Optional, List
import importlib.util

# Add stage2 to path for imports
_stage2_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'stage2'))
if _stage2_path not in sys.path:
    sys.path.insert(0, _stage2_path)

from tensor import Tensor

# Import stage2 nn modules using importlib to avoid name collision
def _load_stage2_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_stage2_path, 'nn', filename))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_module_mod = _load_stage2_module("stage2_module", "module.py")
_linear_mod = _load_stage2_module("stage2_linear", "linear.py")
_norm_mod = _load_stage2_module("stage2_norm", "normalization.py")
_act_mod = _load_stage2_module("stage2_act", "activations.py")

Module = _module_mod.Module
Linear = _linear_mod.Linear
LayerNorm = _norm_mod.LayerNorm
GELU = _act_mod.GELU

# Import local attention module
_attn_path = os.path.dirname(__file__)
if _attn_path not in sys.path:
    sys.path.insert(0, _attn_path)
from attention import MultiHeadAttention


class FeedForward(Module):
    """
    Position-wise Feed-Forward Network.

    FFN(x) = GELU(xW1 + b1)W2 + b2

    Args:
        embed_dim: Input and output dimension
        hidden_dim: Hidden layer dimension (typically 4 * embed_dim)
    """

    def __init__(self, embed_dim: int, hidden_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        self.fc1 = Linear(embed_dim, hidden_dim)
        self.gelu = GELU()
        self.fc2 = Linear(hidden_dim, embed_dim)

        self.register_module('fc1', self.fc1)
        self.register_module('gelu', self.gelu)
        self.register_module('fc2', self.fc2)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim)

        Returns:
            Output tensor of same shape
        """
        batch_size, seq_len, embed_dim = x.shape

        # Reshape to 2D for Linear layers
        x_2d = x.reshape(batch_size * seq_len, embed_dim)

        # FFN: fc1 -> gelu -> fc2
        x_2d = self.fc1(x_2d)
        x_2d = self.gelu(x_2d)
        x_2d = self.fc2(x_2d)

        # Reshape back to 3D
        output = x_2d.reshape(batch_size, seq_len, embed_dim)
        return output

    def __repr__(self):
        return f"FeedForward(embed_dim={self.embed_dim}, hidden_dim={self.hidden_dim})"


class TransformerBlock(Module):
    """
    Single Transformer Decoder Block.

    Pre-norm architecture:
        x = x + Attention(LayerNorm(x))
        x = x + FFN(LayerNorm(x))

    Args:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        ff_hidden_dim: Feed-forward hidden dimension
        eps: LayerNorm epsilon
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ff_hidden_dim: int,
        eps: float = 1e-5
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_hidden_dim = ff_hidden_dim

        # Layer norms (pre-norm architecture)
        self.norm1 = LayerNorm(embed_dim, eps=eps)
        self.norm2 = LayerNorm(embed_dim, eps=eps)

        # Attention
        self.attention = MultiHeadAttention(embed_dim, num_heads)

        # Feed-forward
        self.ffn = FeedForward(embed_dim, ff_hidden_dim)

        # Register all submodules
        self.register_module('norm1', self.norm1)
        self.register_module('norm2', self.norm2)
        self.register_module('attention', self.attention)
        self.register_module('ffn', self.ffn)

    def forward(self, x: Tensor, mask: Optional[np.ndarray] = None) -> Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim)
            mask: Optional causal attention mask

        Returns:
            Output tensor of same shape
        """
        # Pre-norm residual attention
        normed = self.norm1(x)
        attn_out = self.attention(normed, mask=mask)
        x = x + attn_out

        # Pre-norm residual FFN
        normed = self.norm2(x)
        ffn_out = self.ffn(normed)
        x = x + ffn_out

        return x

    def __repr__(self):
        return (f"TransformerBlock(embed_dim={self.embed_dim}, "
                f"num_heads={self.num_heads}, ff_hidden={self.ff_hidden_dim})")


class TransformerDecoder(Module):
    """
    Stack of Transformer Decoder Blocks.

    Args:
        num_layers: Number of transformer blocks
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        ff_hidden_dim: Feed-forward hidden dimension
    """

    def __init__(
        self,
        num_layers: int,
        embed_dim: int,
        num_heads: int,
        ff_hidden_dim: int
    ):
        super().__init__()
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_hidden_dim = ff_hidden_dim

        # Stack of transformer blocks
        self.layers: List[TransformerBlock] = []
        for i in range(num_layers):
            block = TransformerBlock(embed_dim, num_heads, ff_hidden_dim)
            self.layers.append(block)
            self.register_module(f'layer_{i}', block)

        # Final layer norm
        self.final_norm = LayerNorm(embed_dim)
        self.register_module('final_norm', self.final_norm)

    def forward(self, x: Tensor, mask: Optional[np.ndarray] = None) -> Tensor:
        """
        Forward pass through all layers.

        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim)
            mask: Optional causal attention mask

        Returns:
            Output tensor of same shape
        """
        for layer in self.layers:
            x = layer(x, mask=mask)

        x = self.final_norm(x)
        return x

    def __repr__(self):
        return (f"TransformerDecoder(num_layers={self.num_layers}, "
                f"embed_dim={self.embed_dim}, num_heads={self.num_heads})")
