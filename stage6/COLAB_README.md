# Stage 6 on Google Colab (GPU)

The tiled-attention CUDA kernel can't build or run on macOS (no `nvcc` / NVIDIA
GPU). Run it on Colab with a GPU runtime to get the real GPU numbers — the same
way Stage 2's GPU results were produced.

## Steps

1. Colab menu → **Runtime → Change runtime type → GPU** (a T4 is fine).
2. Get the repo into Colab (either clone it, or upload the folder):
   ```
   !git clone <your-repo-url>
   %cd cudatransformer
   ```
3. Run the turnkey script — it installs pybind11, compiles the kernel, checks
   correctness against Stage 3, and benchmarks time + memory:
   ```
   !pip -q install pybind11 numpy
   !python stage6/colab_benchmark.py
   ```
4. It prints the results and writes **`stage6_results_gpu.txt`**. Keep that file
   next to `stage6_results.txt` (like `stage2_results_gpu.txt`), or paste the
   "Tiled GPU (ms)" column into the CUDA column of `stage6_results.txt`.

## What it checks

- **Correctness:** the GPU kernel output must match the Stage 3 NumPy attention
  within `rtol=1e-4, atol=1e-6` (same check as the local run).
- **Time:** tiled CUDA kernel (GPU) vs Stage 3 dense attention (CPU).
- **Memory:** GPU global memory each approach allocates — the tiled kernel never
  allocates the `(B,H,S,S)` score buffer that dense attention needs.

## Manual build (if you'd rather not use the script)

```bash
nvcc -O3 --use_fast_math -shared -Xcompiler -fPIC \
  $(python -m pybind11 --includes) \
  stage6/cuda_ext.cpp stage6/kernels/tiled_attention.cu \
  -o stage6/cuda_ext$(python3-config --extension-suffix)
```
Then in Python: `from stage6 import cuda_ext; cuda_ext.tiled_attention(Q, K, V, causal=True)`
with `Q, K, V` as float32 arrays of shape `(batch, heads, seq, head_dim)`.

## Note on the kernel

`kernels/tiled_attention.cu` uses compile-time `BLOCK_Q = BLOCK_K = 64` and
supports `head_dim <= 128`. Shared memory per block is `2 * BLOCK_K * head_dim *
4` bytes (32 KB at `head_dim=64`), within the 48 KB default. For larger head
dims, lower `BLOCK_K`.
