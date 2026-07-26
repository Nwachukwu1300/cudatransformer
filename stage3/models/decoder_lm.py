"""
Transformer Decoder Language Model.

Full model for next-token prediction.
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

# Import stage2 nn modules using importlib to avoid name collision
def _load_stage2_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_stage2_path, 'nn', filename))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_module_mod = _load_stage2_module("stage2_module", "module.py")
_linear_mod = _load_stage2_module("stage2_linear", "linear.py")

Module = _module_mod.Module
Linear = _linear_mod.Linear

# Import local stage3 nn modules
_stage3_nn_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'nn'))
if _stage3_nn_path not in sys.path:
    sys.path.insert(0, _stage3_nn_path)

from embedding import TokenEmbedding, PositionalEmbedding
from attention import create_causal_mask
from transformer import TransformerDecoder


class DecoderLanguageModel(Module):
    """
    Transformer Decoder Language Model.

    Full model for next-token prediction:
        Token IDs -> Token Embedding + Position Embedding
                  -> Transformer Decoder Stack
                  -> Output Projection
                  -> Logits

    Args:
        vocab_size: Vocabulary size
        max_seq_len: Maximum sequence length
        embed_dim: Embedding dimension
        num_layers: Number of transformer blocks
        num_heads: Number of attention heads
        ff_hidden_dim: Feed-forward hidden dimension (default: 4 * embed_dim)
    """

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        embed_dim: int,
        num_layers: int,
        num_heads: int,
        ff_hidden_dim: Optional[int] = None
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_heads = num_heads

        if ff_hidden_dim is None:
            ff_hidden_dim = 4 * embed_dim
        self.ff_hidden_dim = ff_hidden_dim

        # Embeddings
        self.token_embedding = TokenEmbedding(vocab_size, embed_dim)
        self.pos_embedding = PositionalEmbedding(max_seq_len, embed_dim)

        # Transformer decoder stack
        self.decoder = TransformerDecoder(
            num_layers=num_layers,
            embed_dim=embed_dim,
            num_heads=num_heads,
            ff_hidden_dim=ff_hidden_dim
        )

        # Output projection to vocabulary
        self.output_proj = Linear(embed_dim, vocab_size, bias=False)

        # Register modules
        self.register_module('token_embedding', self.token_embedding)
        self.register_module('pos_embedding', self.pos_embedding)
        self.register_module('decoder', self.decoder)
        self.register_module('output_proj', self.output_proj)

    def forward(self, token_ids: np.ndarray, debug: bool = False) -> Tensor:
        """
        Forward pass.

        Args:
            token_ids: Token indices of shape (batch_size, seq_len)
            debug: If True, print shapes at each step

        Returns:
            Logits tensor of shape (batch_size, seq_len, vocab_size)
        """
        batch_size, seq_len = token_ids.shape

        if debug:
            print(f"  Input token_ids: shape={token_ids.shape}")

        # 1. Token embeddings
        x = self.token_embedding(token_ids)  # (batch, seq, embed_dim)

        if debug:
            print(f"  After token_embedding: shape={x.shape}")

        # 2. Add positional embeddings
        pos = self.pos_embedding(seq_len)  # (seq, embed_dim)

        if debug:
            print(f"  Position embedding: shape={pos.shape}")

        # Broadcast position embedding to batch dimension
        # x: (batch, seq, embed), pos: (seq, embed)
        # Need to make pos (1, seq, embed) for proper broadcasting
        pos_broadcast = pos.reshape(1, seq_len, self.embed_dim)

        # Create tensor that properly broadcasts
        pos_data = np.broadcast_to(pos_broadcast.data, (batch_size, seq_len, self.embed_dim))
        pos_tensor = Tensor(pos_data.copy(),
                           requires_grad=pos.requires_grad,
                           _children=(pos,), _op='broadcast')

        def _backward():
            if pos.requires_grad:
                if pos.grad is None:
                    pos.grad = np.zeros_like(pos.data)
                # Sum gradients across batch dimension
                pos.grad += np.sum(pos_tensor.grad, axis=0)

        pos_tensor._backward = _backward

        x = x + pos_tensor

        if debug:
            print(f"  After adding positions: shape={x.shape}")

        # 3. Create causal mask
        mask = create_causal_mask(seq_len)

        # 4. Transformer decoder
        x = self.decoder(x, mask=mask)

        if debug:
            print(f"  After decoder: shape={x.shape}")

        # 5. Output projection to vocabulary
        # Reshape for linear layer
        x_2d = x.reshape(batch_size * seq_len, self.embed_dim)
        logits_2d = self.output_proj(x_2d)
        logits = logits_2d.reshape(batch_size, seq_len, self.vocab_size)

        if debug:
            print(f"  Output logits: shape={logits.shape}")

        return logits

    def count_parameters(self) -> int:
        """Count total number of trainable parameters."""
        total = 0
        for param in self.parameters():
            total += param.data.size
        return total

    def __repr__(self):
        param_count = self.count_parameters()
        return (f"DecoderLanguageModel(\n"
                f"  vocab_size={self.vocab_size}\n"
                f"  max_seq_len={self.max_seq_len}\n"
                f"  embed_dim={self.embed_dim}\n"
                f"  num_layers={self.num_layers}\n"
                f"  num_heads={self.num_heads}\n"
                f"  ff_hidden_dim={self.ff_hidden_dim}\n"
                f"  Total Parameters: {param_count:,}\n"
                f")")
