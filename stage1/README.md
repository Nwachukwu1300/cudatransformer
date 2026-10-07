# Stage 1: CUDA Kernel Fundamentals

**Goal:** get comfortable writing and testing GPU kernels, and prove each one
is both correct and faster than a naive CPU version.

## What's here

Four hand-written CUDA kernels, each with a CPU reference implementation to
check against and benchmark:

| Kernel | File | Technique |
|---|---|---|
| Vector addition | [`kernels/vector_add.cu`](kernels/vector_add.cu) | One thread per element |
| Matrix multiplication | [`kernels/matmul.cu`](kernels/matmul.cu) | Naive **and** shared-memory tiled versions |
| Softmax | [`kernels/softmax.cu`](kernels/softmax.cu) | One block per row, warp-shuffle reduction |
| Sum reduction | [`kernels/reduce_sum.cu`](kernels/reduce_sum.cu) | Tree reduction + warp shuffle |

See [CUDA_OPTIMIZATIONS.md](../CUDA_OPTIMIZATIONS.md) for a detailed
walkthrough of why each kernel is written the way it is.

[`cuda_ext.cpp`](cuda_ext.cpp) exposes all four to Python via pybind11.
[`kernels/cpu_ops.py`](kernels/cpu_ops.py) is the NumPy reference
implementation each GPU kernel is checked and timed against.

## Deliverable

[`benchmark.py`](benchmark.py) runs every kernel on both CPU and GPU (when
available) across a range of sizes, and writes the results to
`stage1_results.txt`.

```bash
python3 stage1/benchmark.py
```

[`tests/test_kernels.py`](tests/test_kernels.py) checks CPU (and GPU, when
available) output against NumPy references for correctness.

## Results

CPU numbers (every size tested): [`stage1_results.txt`](stage1_results.txt).

**GPU numbers: pending.** This machine has no NVIDIA GPU, so the GPU columns
in a local run show `N/A`. `stage1/colab_benchmark.py` (added in Stage 8) is
ready to produce real numbers on Colab — Stages 2, 6, and 7 all have
Colab-verified GPU numbers, Stage 1 doesn't yet, purely because Colab's free
GPU quota was exhausted (from running Stages 6/7) at the time Stage 8 wrapped
up. Once quota resets, run it and `stage1_results_gpu.txt` will appear at the
repo root (same convention as `stage2_results_gpu.txt`,
`stage6_results_gpu.txt`).

To get real GPU numbers once quota is available:

```bash
# On Colab (GPU runtime):
!git clone https://github.com/Nwachukwu1300/cudatransformer.git
%cd cudatransformer
!pip -q install pybind11 numpy
!python stage1/colab_benchmark.py
```

[`colab_benchmark.py`](colab_benchmark.py) compiles `cuda_ext.cpp` + the four
`.cu` kernels with `nvcc`, then runs the exact same `benchmark.py` so the GPU
numbers use identical shapes and timing harness as the CPU run. Manual build,
if you'd rather not use the script:

```bash
nvcc -O3 --use_fast_math -shared -Xcompiler -fPIC \
  $(python -m pybind11 --includes) \
  stage1/cuda_ext.cpp stage1/kernels/vector_add.cu stage1/kernels/matmul.cu \
  stage1/kernels/softmax.cu stage1/kernels/reduce_sum.cu \
  -o stage1/cuda_ext$(python3-config --extension-suffix)
```

Full numbers for every stage: [BENCHMARKS.md](../BENCHMARKS.md).

### Estimated GPU speedup (not measured — see above)

Until a Colab run fills in real numbers, here's a transparently-labeled
**estimate**, not a measurement, for context:

| Kernel | Expected speedup at largest tested size | Basis |
|---|---|---|
| Matrix multiply (1024³) | roughly 10–30x | Stage 6's tiled-attention kernel — same shared-memory tiling technique as `matmul.cu`'s tiled path — measured **25.7x** at a comparable problem size on a Tesla T4 |
| Vector add / reduce_sum (10M elements) | roughly 5–15x | Large, embarrassingly-parallel elementwise/reduction ops typically see smaller gains than matmul since they're memory-bandwidth-bound rather than compute-bound |
| Softmax (512×2048) | roughly 3–10x | Smaller op; warp-shuffle reduction helps but transfer overhead eats into the win at this size |

**Why not just scale Stage 2's numbers instead:** Stage 2's own GPU run is
real and already in this repo — and it's *slower* than CPU (60.1s vs 27.6s,
see [stage2/README.md](../stage2/README.md)), because its ops are small enough
that per-call Python/pybind11 and host↔device transfer overhead dominates.
That's a real, correctly-measured result for Stage 2's workload, but
extrapolating it to Stage 1 would wrongly predict Stage 1's much larger
matmul/vector ops also run slower on GPU — which contradicts what Stage 6
actually measured for a similarly large, well-parallelized operation. The
estimate above reasons from Stage 6 instead, since it's the closer analogue.

**This table is a placeholder, not a result.** Treat it as a plausible range,
not something to cite as a project outcome — the real numbers from
`stage1/colab_benchmark.py` supersede it the moment that script is run.

## Next

The CPU reference ops here (`cpu_ops.py`) are **not** reused directly by the
autograd engine in Stage 2 — Stage 2 builds its own NumPy `Tensor` class with
its own forward/backward ops, following the same parallelization ideas but
operating within a pure-NumPy training loop. See [stage2/README.md](../stage2/README.md).
