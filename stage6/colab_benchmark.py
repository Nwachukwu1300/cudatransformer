"""
Stage 6 GPU benchmark -- run this ON Google Colab (or any CUDA machine).

It does the part that cannot run on the Mac:
  1. compiles the tiled-attention CUDA kernel with nvcc,
  2. re-checks correctness (GPU kernel vs the Stage 3 NumPy attention),
  3. times the tiled CUDA kernel against the Stage 3 (CPU) attention, and
  4. reports the GPU device memory each approach needs.
Results are printed and written to stage6_results_gpu.txt (same convention as
the existing stage2_results_gpu.txt that was produced on Colab).

Quick start (Colab, GPU runtime):
    !git clone <your repo>            # or upload the folder
    %cd cudatransformer
    !pip -q install pybind11 numpy
    !python stage6/colab_benchmark.py

See stage6/COLAB_README.md for the manual nvcc command if you prefer.
"""

import os
import sys
import sysconfig
import subprocess

import numpy as np

# --------------------------------------------------------------------------
# Reuse everything from the local benchmark so the GPU run tests the SAME
# shapes, tolerance, reference and timing harness -- no duplicated logic.
# --------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import benchmark as bench  # stage6/benchmark.py


def _run(cmd, **kw):
    print("$", cmd if isinstance(cmd, str) else " ".join(cmd))
    return subprocess.run(cmd, shell=isinstance(cmd, str), check=True, **kw)


def gpu_name():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True)
        return out.strip().splitlines()[0]
    except Exception:
        return "unknown GPU"


def build_kernel():
    """Compile stage6/cuda_ext.cpp + kernels/tiled_attention.cu into an
    importable module using nvcc directly (the robust path on Colab)."""
    try:
        import pybind11  # noqa: F401
    except ImportError:
        _run([sys.executable, "-m", "pip", "install", "-q", "pybind11"])

    includes = subprocess.check_output(
        [sys.executable, "-m", "pybind11", "--includes"], text=True).strip()
    ext_suffix = sysconfig.get_config_var("EXT_SUFFIX")  # e.g. .cpython-310-...so
    out_so = os.path.join(_HERE, "cuda_ext" + ext_suffix)

    cpp = os.path.join(_HERE, "cuda_ext.cpp")
    cu = os.path.join(_HERE, "kernels", "tiled_attention.cu")

    cmd = (f"nvcc -O3 --use_fast_math -shared -Xcompiler -fPIC "
           f"{includes} {cpp} {cu} -o {out_so}")
    _run(cmd)

    # Import the freshly built module as a submodule of the stage6 package.
    import importlib
    if "stage6" in sys.modules:
        importlib.invalidate_caches()
    from stage6 import cuda_ext  # noqa
    print(f"Built and imported: {out_so}")
    return cuda_ext


def main():
    print("=" * 70)
    print("Stage 6 GPU Benchmark: Simplified Tiled Attention")
    print("=" * 70)
    dev = gpu_name()
    print("GPU:", dev)

    cuda_ext = build_kernel()

    def run_tiled_gpu(q, k, v, causal):
        return cuda_ext.tiled_attention(q, k, v, causal)

    # ---- 1. Correctness: GPU kernel vs Stage 3 reference ----
    print("\n[1/3] Correctness: tiled CUDA kernel vs Stage 3 attention "
          f"(rtol={bench.RTOL:.0e}, atol={bench.ATOL:.0e})")
    print("-" * 66)
    print(f"{'Shape (B,H,S,d)':>22} {'mask':>6} {'max|diff|':>14} {'result':>8}")
    print("-" * 66)
    corr_rows, all_pass = [], True
    for (b, h, s, d, use_mask) in bench.CORRECTNESS_SHAPES:
        q, k, v = bench._make_inputs(b, h, s, d)
        mask = bench.create_causal_mask(s) if use_mask else None
        ref = bench.run_reference(q, k, v, mask)
        got = run_tiled_gpu(q, k, v, bool(use_mask))
        max_diff = float(np.max(np.abs(ref - got)))
        try:
            np.testing.assert_allclose(got, ref, rtol=bench.RTOL, atol=bench.ATOL)
            passed = True
        except AssertionError:
            passed = False
        all_pass = all_pass and passed
        shape_str = f"({b}, {h}, {s}, {d})"
        mask_str = "causal" if use_mask else "none"
        print(f"{shape_str:>22} {mask_str:>6} {max_diff:>14.3e} "
              f"{('PASS' if passed else 'FAIL'):>8}")
        corr_rows.append((shape_str, mask_str, max_diff, passed))
    print("-" * 66)
    print("Overall:", "PASS" if all_pass else "FAIL")

    # ---- 2. Time: tiled CUDA (GPU) vs Stage 3 dense (CPU) ----
    print(f"\n[2/3] Time: tiled CUDA kernel (GPU) vs Stage 3 dense (CPU)  "
          f"(batch={bench.TIME_BATCH}, heads={bench.TIME_HEADS}, head_dim={bench.TIME_DIM})")
    print("-" * 74)
    print(f"{'SeqLen':>10} {'Stage3 CPU (ms)':>18} {'Tiled GPU (ms)':>18} {'Speedup':>12}")
    print("-" * 74)
    time_rows = []
    for s in bench.TIME_SEQS:
        q, k, v = bench._make_inputs(bench.TIME_BATCH, bench.TIME_HEADS, s, bench.TIME_DIM)
        mask = bench.create_causal_mask(s)
        cpu_ms = bench.benchmark_fn(bench.run_reference, q, k, v, mask)
        gpu_ms = bench.benchmark_fn(run_tiled_gpu, q, k, v, True)
        speedup = f"{cpu_ms / gpu_ms:.2f}x"
        print(f"{s:>10} {cpu_ms:>18.4f} {gpu_ms:>18.4f} {speedup:>12}")
        time_rows.append((s, cpu_ms, gpu_ms, speedup))

    # ---- 3. GPU device memory: what tiling avoids allocating ----
    # The tiled kernel only ever allocates Q,K,V,O in global memory. A dense
    # attention would additionally need the (B,H,S,S) scores AND probs buffers.
    print("\n[3/3] GPU global memory allocated (tiled avoids the O(S^2) score buffer)")
    print("-" * 74)
    print(f"{'SeqLen':>10} {'Tiled (Q,K,V,O)':>18} {'Dense (+ S^2 x2)':>18} {'Extra avoided':>16}")
    print("-" * 74)
    mem_rows = []
    B, H, d = bench.TIME_BATCH, bench.TIME_HEADS, bench.TIME_DIM
    for s in bench.TIME_SEQS:
        tiled_bytes = 4 * B * H * s * d * 4              # Q,K,V,O
        dense_extra = 2 * B * H * s * s * 4              # scores + probs
        dense_bytes = tiled_bytes + dense_extra
        print(f"{s:>10} {bench.format_bytes(tiled_bytes):>18} "
              f"{bench.format_bytes(dense_bytes):>18} "
              f"{bench.format_bytes(dense_extra):>16}")
        mem_rows.append((s, tiled_bytes, dense_bytes, dense_extra))

    _save_gpu_results(dev, corr_rows, all_pass, time_rows, mem_rows)
    print("\nDone. Paste the Tiled GPU column into stage6_results.txt, or keep "
          "stage6_results_gpu.txt alongside it (like stage2_results_gpu.txt).")


def _save_gpu_results(dev, corr_rows, all_pass, time_rows, mem_rows):
    path = os.path.join(_ROOT, "stage6_results_gpu.txt")
    L, bar = [], "=" * 70
    L += [bar, "Stage 6 GPU Benchmark Results: Simplified Tiled Attention",
          "CUDA Transformer Engine - Simplified Tiled Attention (GPU / Colab)", bar, ""]
    L += [f"Device: {dev}", f"NumPy Version: {np.__version__}", ""]

    L += ["-" * 70, "1. Correctness: tiled CUDA kernel vs Stage 3 attention", "-" * 70,
          "Check: np.testing.assert_allclose(rtol=1e-4, atol=1e-6)", ""]
    L.append(f"{'Shape (B,H,S,d)':>22} {'mask':>7} {'max|diff|':>14} {'result':>8}")
    L.append("-" * 60)
    for (shape_str, mask_str, max_diff, passed) in corr_rows:
        L.append(f"{shape_str:>22} {mask_str:>7} {max_diff:>14.3e} "
                 f"{('PASS' if passed else 'FAIL'):>8}")
    L += ["", "Verification:",
          f"  GPU kernel matches Stage 3 within rtol=1e-4: {'YES' if all_pass else 'NO'}", ""]

    L += ["-" * 70, "2. Execution time: tiled CUDA kernel (GPU) vs Stage 3 dense (CPU)",
          "-" * 70,
          f"Config: batch={bench.TIME_BATCH}, heads={bench.TIME_HEADS}, "
          f"head_dim={bench.TIME_DIM}, causal mask",
          "Harness: warmup=3, runs=10 (GPU time includes H2D/D2H transfers)", ""]
    L.append(f"{'SeqLen':>10} {'Stage3 CPU (ms)':>18} {'Tiled GPU (ms)':>18} {'Speedup':>12}")
    L.append("-" * 62)
    for (s, cpu_ms, gpu_ms, speedup) in time_rows:
        L.append(f"{s:>10} {cpu_ms:>18.4f} {gpu_ms:>18.4f} {speedup:>12}")
    L.append("")

    L += ["-" * 70, "3. GPU global memory allocated", "-" * 70,
          "The tiled kernel allocates only Q,K,V,O. A dense attention would also",
          "need the (B,H,S,S) scores and probs buffers -- the O(S^2) cost tiling avoids.",
          ""]
    L.append(f"{'SeqLen':>10} {'Tiled (Q,K,V,O)':>18} {'Dense (+S^2 x2)':>18} {'Extra avoided':>16}")
    L.append("-" * 66)
    for (s, tiled_bytes, dense_bytes, dense_extra) in mem_rows:
        L.append(f"{s:>10} {bench.format_bytes(tiled_bytes):>18} "
                 f"{bench.format_bytes(dense_bytes):>18} "
                 f"{bench.format_bytes(dense_extra):>16}")
    L += ["", bar, "Stage 6 GPU benchmark complete.", bar, ""]

    with open(path, "w") as f:
        f.write("\n".join(L))
    print(f"\nGPU results saved to: {path}")


if __name__ == "__main__":
    main()
