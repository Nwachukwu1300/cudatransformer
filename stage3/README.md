# Stage 3: The Transformer

**Goal:** replace the MLP with a transformer, built on the same Stage 2 engine.

## What's here

[`nn/embedding.py`](nn/embedding.py) — `TokenEmbedding` (learned lookup table,
scatter gradient) and `PositionalEmbedding` (learned, added to token embeddings).

[`nn/attention.py`](nn/attention.py) — `MultiHeadAttention`: Q/K/V projections,
scaled dot-product attention with a causal mask, output projection. Also
`create_causal_mask`, used again in Stage 6's correctness check.

[`nn/transformer.py`](nn/transformer.py) — `TransformerBlock` (pre-norm:
`x = x + Attn(LN(x))`, then `x = x + FFN(LN(x))`) and `TransformerDecoder` (a
stack of blocks + final LayerNorm).

[`models/decoder_lm.py`](models/decoder_lm.py) — `DecoderLanguageModel`, the
full model:

```
token_ids -> token embedding + positional embedding
          -> N transformer blocks (causal attention + FFN, pre-norm residual)
          -> final LayerNorm -> output projection -> logits
```

This exact class, completely unchanged, is reused for both Stage 4 (language
modeling) and Stage 5 (recommendation) — see the root
[README.md](../README.md#architecture) for the diagram and
[stage5_writeup.md](../stage5_writeup.md) for why that works.

Everything here builds on `Tensor`, `Linear`, and `Module` from
[stage2](../stage2/), imported directly — no duplication of the autograd
engine.

## Deliverable

Confirm the transformer runs a full forward and backward pass without errors,
and can do next-token prediction on a small synthetic task.

```bash
python3 train_transformer.py
```

## Results

101,888-param decoder (vocab 10, seq 16, embed 64, 2 layers, 4 heads) trained
on a synthetic repeating-pattern task:

| Metric | Value |
|---|---|
| Initial loss | 0.8842 |
| Final loss (50 epochs) | 0.1349 |
| Final test accuracy | 94.4% |

Full training log, shape checks at every layer, and gradient-check notes:
[`stage3_results.txt`](../stage3_results.txt).

Full numbers for every stage: [BENCHMARKS.md](../BENCHMARKS.md).

## Next

Stage 4 trains this architecture on real text (TinyStories) to prove it
actually learns language, not just a synthetic pattern. See
[stage4/README.md](../stage4/README.md).
