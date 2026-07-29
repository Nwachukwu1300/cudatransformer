"""
Tiled ("block-wise") scaled dot-product attention with online softmax.

This is a plain NumPy reference implementation, written from first principles.
It computes exactly the same result as the Stage 3 attention

    Attention(Q, K, V) = softmax( (Q @ K^T) / sqrt(d) + mask ) @ V

but WITHOUT ever materializing the full (seq_q x seq_k) score/probability
matrix. That single change is the whole point of Stage 6, so the two ideas
that make it work are explained in detail below.

---------------------------------------------------------------------------
WHY TILING AVOIDS THE FULL ATTENTION MATRIX
---------------------------------------------------------------------------
Standard attention builds `scores = Q @ K^T`, an (seq_q x seq_k) matrix, for
every (batch, head). For a sequence of length S that is an S*S block of
floats per head -- memory grows with the SQUARE of the sequence length. At
S = 2048 with 8 batch * 8 heads that is 8*8*2048*2048*4 bytes ~= 1 GB for a
single intermediate, and softmax needs a couple more copies of it.

Tiling refuses to allocate that S*S buffer. Instead we cut K and V into
row-blocks of `block_k` keys and walk over them one at a time. At any moment
we only hold:
  * one small (block_q x block_k) score tile, and
  * three running summaries per query row (see below), each of size O(S).
So peak memory for the attention step drops from O(S^2) to O(S * block).
The catch is that softmax normally needs the WHOLE row at once (to find the
max and the sum), and we are feeding it the row a piece at a time. The online
softmax trick is what lets us do that correctly.

---------------------------------------------------------------------------
THE ONLINE SOFTMAX TRICK
---------------------------------------------------------------------------
A numerically stable softmax over a row of scores s_1..s_N is

    m = max_j s_j                      (row max, for stability)
    p_j = exp(s_j - m)                 (shifted exponentials)
    l   = sum_j p_j                    (normaliser)
    out = sum_j (p_j / l) * v_j        (weighted sum of the value vectors)

The problem: `m` and `l` are reductions over the ENTIRE row, so a naive
implementation needs every score present simultaneously -- i.e. the full
matrix. Online softmax rewrites these reductions so they can be updated
incrementally as each new block of scores arrives, keeping only a running
max `m`, a running normaliser `l`, and a running (un-normalised) output
accumulator `O`.

When a new block gives us a fresh local max `m_blk`, the true running max
becomes `m_new = max(m_old, m_blk)`. Every exponential we accumulated so far
was shifted by the OLD max, so to re-base them onto the new max we multiply
the old running quantities by a correction factor:

    correction = exp(m_old - m_new)          (always in (0, 1])
    l  <- correction * l  +  sum_j exp(s_j - m_new)           over the new block
    O  <- correction * O  +  (exp(s_j - m_new)) @ V_block      over the new block
    m  <- m_new

After the last block, `O / l` is *identical* to the softmax computed on the
whole row at once -- we just never had the whole row in memory. This identity
holds for any block order and any finite scores: if a block is fully masked
(scores ~ -1e9) while the running max is a normal O(1) value, its `correction`
underflows to 0 and it contributes nothing, exactly as it would in the dense
softmax. That is why we can use the same additive -1e9 causal mask as Stage 3
and still match it to floating-point tolerance.
"""

import numpy as np


def tiled_attention(q, k, v, mask=None, block_q=64, block_k=64):
    """Block-wise scaled dot-product attention with online softmax.

    Args:
        q, k, v: float32 arrays of shape (batch, heads, seq_len, head_dim).
        mask:    optional additive mask of shape (seq_q, seq_k) -- the same
                 (seq, seq) array Stage 3 uses, with 0 for "attend" and -1e9
                 for "do not attend". Broadcast across batch and heads.
        block_q: number of query rows processed per tile.
        block_k: number of key/value rows processed per tile.

    Returns:
        float32 array of shape (batch, heads, seq_len, head_dim) -- numerically
        equal (within float tolerance) to the Stage 3 dense attention.
    """
    batch, heads, seq_q, head_dim = q.shape
    seq_k = k.shape[2]

    # Match Stage 3 exactly: scale = 1/sqrt(head_dim), applied AFTER the Q@K^T
    # matmul (see stage3/nn/attention.py). float32 throughout to mirror the
    # reference's precision.
    scale = np.float32(1.0 / np.sqrt(head_dim))

    out = np.zeros_like(q, dtype=np.float32)

    # Iterate every (batch, head) independently -- attention does not mix them.
    # The blocking below is what a single CUDA thread-block will later own.
    for b in range(batch):
        for h in range(heads):
            q_bh = q[b, h]  # (seq_q, head_dim)
            k_bh = k[b, h]  # (seq_k, head_dim)
            v_bh = v[b, h]  # (seq_k, head_dim)

            # ----- outer loop over blocks of QUERY rows -----
            for qs in range(0, seq_q, block_q):
                qe = min(qs + block_q, seq_q)
                q_tile = q_bh[qs:qe]              # (bq, head_dim)
                bq = qe - qs

                # Running summaries for this query block. One entry per query
                # row; these are the ONLY state we carry across key blocks, and
                # they are O(seq) not O(seq^2).
                m_run = np.full((bq, 1), -np.inf, dtype=np.float32)  # running max
                l_run = np.zeros((bq, 1), dtype=np.float32)          # running sum
                o_run = np.zeros((bq, head_dim), dtype=np.float32)   # running output

                # ----- inner loop over blocks of KEY/VALUE rows -----
                for ks in range(0, seq_k, block_k):
                    ke = min(ks + block_k, seq_k)
                    k_tile = k_bh[ks:ke]         # (bk, head_dim)
                    v_tile = v_bh[ks:ke]         # (bk, head_dim)

                    # Score tile for just this (query block x key block).
                    # This (bq x bk) tile is the largest thing we allocate --
                    # it is never assembled into the full (seq_q x seq_k) matrix.
                    scores = (q_tile @ k_tile.T) * scale     # (bq, bk)
                    if mask is not None:
                        scores = scores + mask[qs:qe, ks:ke]

                    # Local (per-row) max over just this block's keys.
                    m_blk = np.max(scores, axis=1, keepdims=True)     # (bq, 1)

                    # New running max, and the correction that re-bases every
                    # exponential accumulated so far onto that new max.
                    m_new = np.maximum(m_run, m_blk)                  # (bq, 1)
                    correction = np.exp(m_run - m_new)               # (bq, 1)

                    # Shifted exponentials for this block, using the new max.
                    p = np.exp(scores - m_new)                       # (bq, bk)

                    # Update the running normaliser and output accumulator.
                    # correction rescales the old contributions; the new block
                    # is added on top. (p @ v_tile) is the block's weighted
                    # value sum -- (bq x bk) @ (bk x head_dim) -> (bq x head_dim).
                    l_run = correction * l_run + np.sum(p, axis=1, keepdims=True)
                    o_run = correction * o_run + p @ v_tile
                    m_run = m_new

                # Normalise once at the end: divide the accumulated weighted
                # values by the accumulated softmax denominator. Every causal
                # row has at least its own position unmasked, so l_run > 0.
                out[b, h, qs:qe] = o_run / l_run

    return out
