"""
Flax reimplementation of the Stage 3 transformer (decoder-only language model).

This is an ORIGINAL Flax/linen module written to mirror the Stage 3 architecture
exactly -- it is NOT a port of the CUDA kernels or the NumPy autograd engine. The
point of Stage 7 is a fair GPU-vs-TPU comparison, so every architectural detail
below is matched to stage3/ (see the mapping notes inline).

Architecture (Stage 4 config: vocab 2000, seq 64, embed 128, 4 layers, 4 heads,
FFN 512) -> 1,311,488 parameters, identical to the Stage 3/4 model.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import flax.linen as nn


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 2000       # Stage 4 tokenizer vocab
    max_seq_len: int = 64        # Stage 4 context length
    embed_dim: int = 128
    num_layers: int = 4
    num_heads: int = 4           # head_dim = 128/4 = 32
    ff_hidden_dim: int = 512     # 4 * embed_dim


# Init schemes matched to Stage 3:
#   - Linear (nn.Dense) kernels: He/Kaiming normal, std = sqrt(2/fan_in)
#     (stage2/nn/linear.py). Flax's variance_scaling(2.0, fan_in, normal) is the
#     untruncated He-normal, matching np.randn * sqrt(2/in_features).
#   - Embeddings: normal with std = sqrt(1/embed_dim) (stage3/nn/embedding.py).
_DENSE_KERNEL_INIT = nn.initializers.variance_scaling(2.0, "fan_in", "normal")


def _embed_init(embed_dim):
    return nn.initializers.normal(stddev=(1.0 / embed_dim) ** 0.5)


class MultiHeadAttention(nn.Module):
    """Causal scaled-dot-product attention, matching stage3/nn/attention.py.

    Q/K/V/O are bias-free linear projections; scale = 1/sqrt(head_dim); an
    additive -1e9 causal mask (strict upper triangle) is applied before a
    softmax over the key axis.
    """
    embed_dim: int
    num_heads: int

    @nn.compact
    def __call__(self, x):
        B, S, E = x.shape
        H = self.num_heads
        hd = E // H
        scale = 1.0 / jnp.sqrt(jnp.float32(hd))

        # Bias-free projections (stage3 uses bias=False for W_q/W_k/W_v/W_o).
        q = nn.Dense(E, use_bias=False, kernel_init=_DENSE_KERNEL_INIT, name="q")(x)
        k = nn.Dense(E, use_bias=False, kernel_init=_DENSE_KERNEL_INIT, name="k")(x)
        v = nn.Dense(E, use_bias=False, kernel_init=_DENSE_KERNEL_INIT, name="v")(x)

        # (B, S, E) -> (B, S, H, hd); attention is computed per head.
        q = q.reshape(B, S, H, hd)
        k = k.reshape(B, S, H, hd)
        v = v.reshape(B, S, H, hd)

        # scores[b,h,i,j] = scale * sum_d q[b,i,h,d] * k[b,j,h,d]
        scores = jnp.einsum("bihd,bjhd->bhij", q, k) * scale  # (B, H, S, S)

        # Additive causal mask: position i may attend to j <= i. Strict upper
        # triangle gets -1e9 (matches stage3 create_causal_mask), so it never
        # materializes anything larger than (S, S).
        causal = jnp.tril(jnp.ones((S, S), dtype=bool))
        scores = jnp.where(causal[None, None, :, :], scores, jnp.float32(-1e9))

        attn = jax.nn.softmax(scores, axis=-1)                # over keys
        out = jnp.einsum("bhij,bjhd->bihd", attn, v)          # (B, S, H, hd)
        out = out.reshape(B, S, E)

        return nn.Dense(E, use_bias=False, kernel_init=_DENSE_KERNEL_INIT, name="o")(out)


class FeedForward(nn.Module):
    """Position-wise FFN: Dense(E->4E) -> GELU(tanh approx) -> Dense(4E->E).

    Both Dense layers HAVE bias (stage3 FFN uses bias=True), and the activation
    is the tanh approximation of GELU (stage2/nn/activations.py), i.e.
    nn.gelu(..., approximate=True).
    """
    embed_dim: int
    ff_hidden_dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.ff_hidden_dim, kernel_init=_DENSE_KERNEL_INIT, name="fc1")(x)
        x = nn.gelu(x, approximate=True)
        x = nn.Dense(self.embed_dim, kernel_init=_DENSE_KERNEL_INIT, name="fc2")(x)
        return x


class TransformerBlock(nn.Module):
    """Pre-norm block: x = x + Attn(LN(x)); x = x + FFN(LN(x)).

    The residual is added to the raw (un-normalized) input, matching
    stage3/nn/transformer.py. LayerNorm uses eps=1e-5 (Stage 3 value) and Flax's
    default biased variance.
    """
    embed_dim: int
    num_heads: int
    ff_hidden_dim: int

    @nn.compact
    def __call__(self, x):
        h = x + MultiHeadAttention(self.embed_dim, self.num_heads, name="attn")(
            nn.LayerNorm(epsilon=1e-5, name="ln1")(x))
        out = h + FeedForward(self.embed_dim, self.ff_hidden_dim, name="ffn")(
            nn.LayerNorm(epsilon=1e-5, name="ln2")(h))
        return out


class DecoderLM(nn.Module):
    """Decoder-only LM mirroring stage3/models/decoder_lm.py.

    token_embed + learned positional_embed (added, NO sqrt(d) scaling)
      -> N pre-norm transformer blocks
      -> final LayerNorm
      -> untied, bias-free output projection to vocab logits.
    """
    config: ModelConfig

    @nn.compact
    def __call__(self, token_ids):
        cfg = self.config
        B, S = token_ids.shape

        tok = nn.Embed(cfg.vocab_size, cfg.embed_dim,
                       embedding_init=_embed_init(cfg.embed_dim), name="token_embed")(token_ids)
        pos_ids = jnp.arange(S)
        pos = nn.Embed(cfg.max_seq_len, cfg.embed_dim,
                       embedding_init=_embed_init(cfg.embed_dim), name="pos_embed")(pos_ids)
        x = tok + pos[None, :, :]  # broadcast over batch; no scaling of embeddings

        for i in range(cfg.num_layers):
            x = TransformerBlock(cfg.embed_dim, cfg.num_heads, cfg.ff_hidden_dim,
                                 name=f"block_{i}")(x)

        x = nn.LayerNorm(epsilon=1e-5, name="final_ln")(x)

        # Untied output head, no bias (separate from the token embedding matrix).
        logits = nn.Dense(cfg.vocab_size, use_bias=False,
                          kernel_init=_DENSE_KERNEL_INIT, name="output")(x)
        return logits  # (B, S, vocab_size)


def count_params(params) -> int:
    """Total number of scalar parameters in a Flax params pytree."""
    return int(sum(x.size for x in jax.tree_util.tree_leaves(params)))
