"""
Stage 1 GPU benchmark -- run this ON Google Colab (or any CUDA machine).

stage1/benchmark.py already benchmarks GPU vs CPU automatically -- it just
needs `stage1.cuda_ext` to be importable, which requires compiling
cuda_ext.cpp + the four kernels/*.cu files with nvcc. This script does that
compile step, then calls the existing benchmark unchanged, so the GPU run
uses the exact same shapes, timing harness and output format as the local
CPU-only run (same convention as stage6/colab_benchmark.py).

Quick start (Colab, GPU runtime):
    !git clone <your repo>            # or upload the folder
    %cd cudatransformer
    !pip -q install pybind11 numpy
    !python stage1/colab_benchmark.py

See stage1/README.md for the manual nvcc command if you prefer.
"""

import os
import subprocess
import sys
import sysconfig

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)


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
    """Compile stage1/cuda_ext.cpp + kernels/*.cu into an importable module
    using nvcc directly (the robust path on Colab, same as stage6)."""
    try:
        import pybind11  # noqa: F401
    except ImportError:
        _run([sys.executable, "-m", "pip", "install", "-q", "pybind11"])

    includes = subprocess.check_output(
        [sys.executable, "-m", "pybind11", "--includes"], text=True).strip()
    ext_suffix = sysconfig.get_config_var("EXT_SUFFIX")
    out_so = os.path.join(_HERE, "cuda_ext" + ext_suffix)

    cpp = os.path.join(_HERE, "cuda_ext.cpp")
    kernels = os.path.join(_HERE, "kernels")
    cu_files = " ".join(
        os.path.join(kernels, f)
        for f in ("vector_add.cu", "matmul.cu", "softmax.cu", "reduce_sum.cu")
    )

    cmd = (f"nvcc -O3 --use_fast_math -shared -Xcompiler -fPIC "
           f"{includes} {cpp} {cu_files} -o {out_so}")
    _run(cmd)

    import importlib
    if "stage1" in sys.modules:
        importlib.invalidate_caches()
    from stage1 import cuda_ext  # noqa
    print(f"Built and imported: {out_so}")
    return cuda_ext


def main():
    print("=" * 70)
    print("Stage 1 GPU Benchmark: Kernel Fundamentals")
    print("=" * 70)
    dev = gpu_name()
    print("GPU:", dev)

    build_kernel()  # import side effect: stage1.benchmark will now see it

    # Re-run the existing benchmark module fresh so its CUDA_AVAILABLE check
    # (which happens at import time) picks up the module we just built.
    import importlib
    if "stage1.benchmark" in sys.modules:
        del sys.modules["stage1.benchmark"]
    if "benchmark" in sys.modules:
        del sys.modules["benchmark"]
    from stage1 import benchmark as bench

    print(f"\nCUDA Available: {bench.CUDA_AVAILABLE}")
    assert bench.CUDA_AVAILABLE, "cuda_ext did not import after building -- check the nvcc build log above"

    all_results = []
    all_results.extend(bench.benchmark_vector_add())
    all_results.extend(bench.benchmark_matmul())
    all_results.extend(bench.benchmark_softmax())
    all_results.extend(bench.benchmark_reduce_sum())

    print("\n" + "=" * 70)

    # Write to stage1_results_gpu.txt at the repo root (same convention as
    # stage2_results_gpu.txt / stage6_results_gpu.txt), not the CPU file.
    _save_gpu_results(all_results, dev)
    print("\nDone. GPU benchmark complete.")


def _save_gpu_results(results, dev):
    import numpy as np
    from stage1.benchmark import format_speedup

    path = os.path.join(_ROOT, "stage1_results_gpu.txt")
    L, bar = [], "=" * 70
    L += [bar, "Stage 1 GPU Benchmark Results", "CUDA Transformer Engine - Kernel Fundamentals (GPU / Colab)", bar, ""]
    L += [f"Device: {dev}", f"NumPy Version: {np.__version__}", ""]

    operations = ["vector_add", "matmul", "softmax", "reduce_sum"]
    op_names = {
        "vector_add": "Vector Addition",
        "matmul": "Matrix Multiplication",
        "softmax": "Softmax",
        "reduce_sum": "Sum Reduction",
    }
    for op in operations:
        op_results = [r for r in results if r["operation"] == op]
        if not op_results:
            continue
        L.append(f"\n{op_names[op]}:")
        L.append("-" * 60)
        L.append(f"{'Size':>15} {'CPU (ms)':>15} {'GPU (ms)':>15} {'Speedup':>12}")
        L.append("-" * 60)
        for r in op_results:
            cpu_str = f"{r['cpu_ms']:.4f}"
            gpu_str = f"{r['gpu_ms']:.4f}" if r["gpu_ms"] else "N/A"
            speedup = format_speedup(r["cpu_ms"], r["gpu_ms"])
            L.append(f"{str(r['size']):>15} {cpu_str:>15} {gpu_str:>15} {speedup:>12}")
        L.append("")

    L += ["", bar, "Stage 1 GPU benchmark complete.", bar, ""]

    with open(path, "w") as f:
        f.write("\n".join(L))
    print(f"\nGPU results saved to: {path}")


if __name__ == "__main__":
    main()
