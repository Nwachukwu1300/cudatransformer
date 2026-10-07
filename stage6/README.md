# Stage 6: Simplified Tiled Attention

**Goal:** replace standard attention with a simplified tiled version that
reduces memory reads and writes, in the spirit of FlashAttention — not a full
production implementation, but a working version with a clear writeup.

## What's here

[`tiled_attention.py`](tiled_attention.py) — a from-scratch NumPy
implementation of tiled attention with **online softmax**: processes the
key/value sequence in blocks and keeps running max/sum/accumulator values per
query row, so the full `(seq × seq)` score matrix is never materialized — see
[CUDA_OPTIMIZATIONS.md](../CUDA_OPTIMIZATIONS.md#6-fusing-an-entire-operator-into-one-kernel-pass-stage-6-tiled-attention)
for how the online-softmax math works.

[`kernels/tiled_attention.cu`](kernels/tiled_attention.cu) — the same
algorithm as a real CUDA kernel: shared-memory K/V tiles, fused
matmul→softmax→matmul in one kernel launch, written to the Stage 1/2
conventions.

[`benchmark.py`](benchmark.py) — correctness (vs Stage 3's dense attention),
CPU timing, and peak-memory comparison, all running locally.

[`colab_benchmark.py`](colab_benchmark.py) — compiles and runs the CUDA kernel
on a real GPU; see [COLAB_README.md](COLAB_README.md).

## Deliverable

```bash
# Local: correctness + CPU timing + memory (no GPU needed)
python3 stage6/benchmark.py
```

```bash
# Colab GPU: compiles and benchmarks the real CUDA kernel
!git clone https://github.com/Nwachukwu1300/cudatransformer.git
%cd cudatransformer
!pip -q install pybind11 numpy
!python stage6/colab_benchmark.py
```
Full steps: [COLAB_README.md](COLAB_README.md).

## Results

**Correctness:** verified numerically identical to Stage 3's dense attention
(`stage3/nn/attention.py`) — max|diff| ≈ 5×10⁻⁷ across all tested shapes, both
in NumPy and in the compiled CUDA kernel.

**Speed** — tiled CUDA kernel (Tesla T4) vs Stage 3 dense attention (CPU):

| Seq len | Stage 3 CPU | Tiled GPU | Speedup |
|---|---|---|---|
| 64 | 2.15 ms | 0.86 ms | 2.50x |
| 1024 | 588.55 ms | 22.89 ms | **25.71x** |

**Memory:** at seq=1024, the tiled kernel avoids allocating **128 MB** of
`(batch, heads, seq, seq)` score/probability buffer that dense attention needs.

Full tables: [`stage6_results.txt`](../stage6_results.txt) (CPU),
[`stage6_results_gpu.txt`](../stage6_results_gpu.txt) (GPU).
Full numbers for every stage: [BENCHMARKS.md](../BENCHMARKS.md).

## Note

This is forward-only and **not wired into the training path** in Stages 3–5 —
it's a standalone, benchmarked proof that a faster, more memory-efficient
attention implementation is possible without changing the math, verified
against the live model's own attention output.

## Next

Stage 7 compares training this architecture on a GPU vs a TPU. See
[stage7/README.md](../stage7/README.md).
