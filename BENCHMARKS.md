# Benchmarks

Every measured number in this project, by stage. Raw output lives in the
`*_results*.txt` files at the repo root (and `stage1/stage1_results.txt`); this
is the readable summary.

Unless noted, CPU numbers were measured on the development machine (Apple
Silicon Mac, no NVIDIA GPU) and GPU/TPU numbers on Google Colab.

## Stage 1 — CUDA kernel fundamentals

CPU vs GPU for four hand-written kernels. CPU: local Mac (NumPy). GPU: Colab
(see [stage1/stage1_results_gpu.txt](stage1_results_gpu.txt) once run — see
[stage1/README.md](stage1/README.md) for how).

Local CPU numbers (`stage1/stage1_results.txt`):

| Operation | Size | CPU time |
|---|---|---|
| Vector add | 10,000,000 | 2.30 ms |
| Matmul | 1024×1024×1024 | 10.17 ms |
| Softmax | 512×2048 | 3.73 ms |
| Sum reduction | 10,000,000 | 1.73 ms |

GPU numbers: see `stage1_results_gpu.txt` (produced by
`stage1/colab_benchmark.py` on Colab — added in Stage 8 to close a gap where
Stage 1 had never been run on real GPU hardware, unlike every later stage).

## Stage 2 — Autograd engine + MNIST MLP

3-layer MLP (784 → 256 → 128 → 10, 235,146 params), Adam, batch 64, 10 epochs,
trained with hand-written kernels/autograd only.

| Run | Best test accuracy | Final train loss | Training time |
|---|---|---|---|
| CPU (local) | 98.00% | 0.0144 | 27.6s |
| GPU (Colab) | 98.14% | 0.0148 | 60.1s |

(GPU time includes the same per-kernel-call Python/pybind11 overhead the CPU
run doesn't pay for small ops — not a speed claim, a correctness + parity
check that the same math runs on both paths.)

## Stage 3 — Transformer decoder (synthetic task)

101,888-param decoder (vocab 10, seq 16, embed 64, 2 layers, 4 heads), trained
on a synthetic repeating-pattern task to confirm a full forward + backward pass
through attention works correctly.

| Metric | Value |
|---|---|
| Initial loss | 0.8842 |
| Final loss (50 epochs) | 0.1349 |
| Final test accuracy | 94.4% |
| Training time | 27.8s |

## Stage 4 — Tiny language model (TinyStories)

1,311,488-param decoder (vocab 2,000, seq 64, embed 128, 4 layers, 4 heads),
first 5MB of TinyStories, 15 epochs, CPU-only (NumPy engine).

| Metric | Value |
|---|---|
| Train loss | 3.40 → **1.69** |
| Val loss | 2.87 → 2.66 (best 2.45, epoch 6) |
| Training time | 7,442.3s (~2.07 hr) |

Example completion: *"The king walked into"* → *"the king walked into the
forest. he was not scared of the lion. he was too scared to go away. the king
saw the queen and wanted the best."* — full set in
[stage4_results.txt](stage4_results.txt).

## Stage 5 — Sequential recommender (MovieLens 1M)

Same `DecoderLanguageModel` class, same hyperparameters as Stage 4 — only the
vocabulary (3,710 movie IDs) and data segmentation (per-user sequences) change.
1,749,248 params (bigger vocab = bigger embedding table). 6,040 users, 1.0M
interactions, 15 epochs, CPU-only.

| Metric | Value |
|---|---|
| Train loss | 6.46 → **4.20** |
| Val loss | 5.77 → 5.03 |
| Training time | 6,536.8s (~1.82 hr) |

Example: a user who watched *A Bug's Life, Antz, The Hunchback of Notre Dame,
Hercules, Mulan* → top prediction *The Lion King* (score 11.68); actual next
movie was *Pocahontas*. Full predictions in
[stage5_results.txt](stage5_results.txt); the structural argument for why this
is the same task as Stage 4 is in [stage5_writeup.md](stage5_writeup.md).

## Stage 6 — Simplified tiled attention (FlashAttention-style)

Original from-scratch tiled attention with online softmax — never materializes
the O(seq²) score matrix. Verified numerically identical to Stage 3's dense
attention (max|diff| ≈ 5×10⁻⁷ across all tested shapes), then benchmarked on a
Colab Tesla T4.

**Correctness** (`stage6_results.txt`, `stage6_results_gpu.txt`): all 5 tested
shapes PASS within `rtol=1e-4, atol=1e-6`, both on CPU (Python tiled vs Stage 3)
and GPU (CUDA kernel vs Stage 3).

**Speed** — tiled CUDA kernel (GPU) vs Stage 3 dense attention (CPU):

| Seq len | Stage 3 CPU | Tiled GPU | Speedup |
|---|---|---|---|
| 64 | 2.15 ms | 0.86 ms | 2.50x |
| 128 | 8.54 ms | 1.70 ms | 5.04x |
| 256 | 30.69 ms | 3.90 ms | 7.87x |
| 512 | 117.05 ms | 10.39 ms | 11.26x |
| 1024 | 588.55 ms | 22.89 ms | **25.71x** |

**Memory** — GPU global memory allocated (tiled avoids the O(seq²) score buffer):

| Seq len | Tiled (Q,K,V,O) | Dense (+ scores) | Extra avoided |
|---|---|---|---|
| 64 | 1.0 MB | 1.5 MB | 512 KB |
| 256 | 4.0 MB | 12.0 MB | 8 MB |
| 1024 | 16.0 MB | 144.0 MB | **128 MB** |

## Stage 7 — GPU vs TPU training comparison (JAX/Flax)

A Flax reimplementation of the exact Stage 3 architecture (1,311,488 params,
same config as Stage 4), trained on the same TinyStories recipe, so only the
hardware changes between rows.

| Device | Time | Steps/s | Peak mem | Est. cost | Train loss | Val loss |
|---|---|---|---|---|---|---|
| CPU (local smoke, 100 steps) | 3.3s | 30.1 | — | — | 4.987 | 4.935 |
| Tesla T4 (Colab GPU) | 149.6s | 221.3 | 140.2 MB | $0.0145 | 1.705 | 2.636 |
| **TPU v5e** (Colab TPU) | **58.0s** | **570.8** | 90.8 MB | $0.0193 | 1.705 | 2.638 |
| CPU (Stage 4 NumPy engine, reference) | 124.0 min | — | — | — | 1.693 | 2.656 |

TPU was **2.6x faster than the T4 GPU** and **~128x faster than our own
from-scratch CPU engine**, while reaching essentially the same loss as the
Stage 4 target (confirms the reimplementation is faithful to the original
architecture). Caveat from `stage7_results.txt`: at 1.3M params this shows
mechanics and relative speed, not TPU's large-scale advantage — the point was
a fair, identical-code comparison, not a scaling study.

## Summary: fastest attention path at seq_len=1024

For context, here's every attention implementation in the repo at the same
problem size:

| Implementation | Where | Time (seq=1024) |
|---|---|---|
| Stage 3 dense attention | CPU (NumPy) | 588.5 ms |
| Stage 6 tiled attention | CPU (Python, same tiling logic) | 165.5 ms |
| Stage 6 tiled attention | **GPU (CUDA kernel, Tesla T4)** | **22.9 ms** |

Going from naive dense CPU attention to the hand-written tiled CUDA kernel is a
**25.7x** wall-clock improvement at this sequence length, achieved without
changing the math — only the memory access pattern.
