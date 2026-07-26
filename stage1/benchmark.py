"""
Stage 1 Benchmark Script

Benchmarks CPU and GPU implementations of:
- Vector addition
- Matrix multiplication
- Softmax
- Sum reduction

Outputs a formatted table and saves results to stage1_results.txt
"""

import numpy as np
import time
import sys
import os
from typing import Callable, Tuple, List

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from stage1.kernels import cpu_ops

# Try to import CUDA extension
try:
    from stage1 import cuda_ext as gpu_ops
    CUDA_AVAILABLE = True
except ImportError:
    gpu_ops = None
    CUDA_AVAILABLE = False


def benchmark_fn(fn: Callable, *args, warmup: int = 3, runs: int = 10) -> float:
    """
    Benchmark a function and return average time in milliseconds.

    Args:
        fn: Function to benchmark
        *args: Arguments to pass to function
        warmup: Number of warmup runs (not timed)
        runs: Number of timed runs

    Returns:
        Average execution time in milliseconds
    """
    # Warmup
    for _ in range(warmup):
        fn(*args)

    # Timed runs
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        fn(*args)
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms

    return np.mean(times)


def format_time(ms: float) -> str:
    """Format time in ms with appropriate precision."""
    if ms < 0.01:
        return f"{ms*1000:.2f}us"
    elif ms < 1:
        return f"{ms:.3f}ms"
    elif ms < 1000:
        return f"{ms:.1f}ms"
    else:
        return f"{ms/1000:.2f}s"


def format_speedup(cpu_time: float, gpu_time: float) -> str:
    """Format speedup ratio."""
    if gpu_time is None:
        return "N/A"
    speedup = cpu_time / gpu_time
    return f"{speedup:.1f}x"


def benchmark_vector_add() -> List[dict]:
    """Benchmark vector addition across different sizes."""
    sizes = [1_000, 10_000, 100_000, 1_000_000, 10_000_000]
    results = []

    print("\nVector Addition:")
    print("-" * 60)
    print(f"{'Size':>12} {'CPU':>12} {'GPU':>12} {'Speedup':>12}")
    print("-" * 60)

    for size in sizes:
        np.random.seed(42)
        a = np.random.randn(size).astype(np.float32)
        b = np.random.randn(size).astype(np.float32)

        cpu_time = benchmark_fn(cpu_ops.vector_add, a, b)

        if CUDA_AVAILABLE:
            gpu_time = benchmark_fn(gpu_ops.vector_add, a, b)
        else:
            gpu_time = None

        speedup = format_speedup(cpu_time, gpu_time)
        gpu_str = format_time(gpu_time) if gpu_time else "N/A"

        print(f"{size:>12,} {format_time(cpu_time):>12} {gpu_str:>12} {speedup:>12}")

        results.append({
            'operation': 'vector_add',
            'size': size,
            'cpu_ms': cpu_time,
            'gpu_ms': gpu_time
        })

    return results


def benchmark_matmul() -> List[dict]:
    """Benchmark matrix multiplication across different sizes."""
    sizes = [(64, 64), (128, 128), (256, 256), (512, 512), (1024, 1024)]
    results = []

    print("\nMatrix Multiplication:")
    print("-" * 60)
    print(f"{'Size':>12} {'CPU':>12} {'GPU':>12} {'Speedup':>12}")
    print("-" * 60)

    for M, N in sizes:
        K = M  # Square matrices
        np.random.seed(42)
        A = np.random.randn(M, K).astype(np.float32)
        B = np.random.randn(K, N).astype(np.float32)

        cpu_time = benchmark_fn(cpu_ops.matmul, A, B)

        if CUDA_AVAILABLE:
            gpu_time = benchmark_fn(gpu_ops.matmul, A, B)
        else:
            gpu_time = None

        speedup = format_speedup(cpu_time, gpu_time)
        gpu_str = format_time(gpu_time) if gpu_time else "N/A"
        size_str = f"{M}x{N}"

        print(f"{size_str:>12} {format_time(cpu_time):>12} {gpu_str:>12} {speedup:>12}")

        results.append({
            'operation': 'matmul',
            'size': f"{M}x{K}x{N}",
            'cpu_ms': cpu_time,
            'gpu_ms': gpu_time
        })

    return results


def benchmark_softmax() -> List[dict]:
    """Benchmark softmax across different sizes."""
    sizes = [(32, 128), (64, 256), (128, 512), (256, 1024), (512, 2048)]
    results = []

    print("\nSoftmax:")
    print("-" * 60)
    print(f"{'Size':>12} {'CPU':>12} {'GPU':>12} {'Speedup':>12}")
    print("-" * 60)

    for rows, cols in sizes:
        np.random.seed(42)
        x = np.random.randn(rows, cols).astype(np.float32)

        cpu_time = benchmark_fn(lambda x: cpu_ops.softmax(x, axis=-1), x)

        if CUDA_AVAILABLE:
            gpu_time = benchmark_fn(gpu_ops.softmax, x)
        else:
            gpu_time = None

        speedup = format_speedup(cpu_time, gpu_time)
        gpu_str = format_time(gpu_time) if gpu_time else "N/A"
        size_str = f"{rows}x{cols}"

        print(f"{size_str:>12} {format_time(cpu_time):>12} {gpu_str:>12} {speedup:>12}")

        results.append({
            'operation': 'softmax',
            'size': f"{rows}x{cols}",
            'cpu_ms': cpu_time,
            'gpu_ms': gpu_time
        })

    return results


def benchmark_reduce_sum() -> List[dict]:
    """Benchmark sum reduction across different sizes."""
    sizes = [1_000, 10_000, 100_000, 1_000_000, 10_000_000]
    results = []

    print("\nSum Reduction:")
    print("-" * 60)
    print(f"{'Size':>12} {'CPU':>12} {'GPU':>12} {'Speedup':>12}")
    print("-" * 60)

    for size in sizes:
        np.random.seed(42)
        x = np.random.randn(size).astype(np.float32)

        cpu_time = benchmark_fn(cpu_ops.reduce_sum, x)

        if CUDA_AVAILABLE:
            gpu_time = benchmark_fn(gpu_ops.reduce_sum, x)
        else:
            gpu_time = None

        speedup = format_speedup(cpu_time, gpu_time)
        gpu_str = format_time(gpu_time) if gpu_time else "N/A"

        print(f"{size:>12,} {format_time(cpu_time):>12} {gpu_str:>12} {speedup:>12}")

        results.append({
            'operation': 'reduce_sum',
            'size': size,
            'cpu_ms': cpu_time,
            'gpu_ms': gpu_time
        })

    return results


def save_results(results: List[dict], filename: str):
    """Save benchmark results to a file."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(script_dir, filename)

    with open(filepath, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("Stage 1 Benchmark Results\n")
        f.write("CUDA Transformer Engine - Kernel Fundamentals\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"CUDA Available: {CUDA_AVAILABLE}\n")
        f.write(f"NumPy Version: {np.__version__}\n\n")

        # Group by operation
        operations = ['vector_add', 'matmul', 'softmax', 'reduce_sum']
        op_names = {
            'vector_add': 'Vector Addition',
            'matmul': 'Matrix Multiplication',
            'softmax': 'Softmax',
            'reduce_sum': 'Sum Reduction'
        }

        for op in operations:
            op_results = [r for r in results if r['operation'] == op]
            if not op_results:
                continue

            f.write(f"\n{op_names[op]}:\n")
            f.write("-" * 60 + "\n")
            f.write(f"{'Size':>15} {'CPU (ms)':>15} {'GPU (ms)':>15} {'Speedup':>12}\n")
            f.write("-" * 60 + "\n")

            for r in op_results:
                cpu_str = f"{r['cpu_ms']:.4f}"
                gpu_str = f"{r['gpu_ms']:.4f}" if r['gpu_ms'] else "N/A"
                speedup = format_speedup(r['cpu_ms'], r['gpu_ms'])
                size_str = str(r['size'])
                f.write(f"{size_str:>15} {cpu_str:>15} {gpu_str:>15} {speedup:>12}\n")

            f.write("\n")

        f.write("\n" + "=" * 70 + "\n")
        if not CUDA_AVAILABLE:
            f.write("Note: GPU benchmarks will be available when running on a CUDA-enabled system.\n")
        f.write("=" * 70 + "\n")

    print(f"\nResults saved to: {filepath}")


def main():
    print("=" * 70)
    print("Stage 1 Benchmark: CUDA Kernel Fundamentals")
    print("=" * 70)
    print(f"\nCUDA Available: {CUDA_AVAILABLE}")
    if not CUDA_AVAILABLE:
        print("(GPU columns will show N/A - running CPU-only benchmarks)")

    all_results = []

    all_results.extend(benchmark_vector_add())
    all_results.extend(benchmark_matmul())
    all_results.extend(benchmark_softmax())
    all_results.extend(benchmark_reduce_sum())

    print("\n" + "=" * 70)

    save_results(all_results, 'stage1_results.txt')

    print("\nBenchmark complete!")


if __name__ == '__main__':
    main()
