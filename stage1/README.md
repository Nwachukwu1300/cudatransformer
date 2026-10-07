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
GPU numbers (Colab Tesla T4): `stage1_results_gpu.txt` at the repo root.

This machine has no NVIDIA GPU, so the GPU columns in a local run show `N/A`.
To get real GPU numbers:

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

## Next

The CPU reference ops here (`cpu_ops.py`) are **not** reused directly by the
autograd engine in Stage 2 — Stage 2 builds its own NumPy `Tensor` class with
its own forward/backward ops, following the same parallelization ideas but
operating within a pure-NumPy training loop. See [stage2/README.md](../stage2/README.md).
