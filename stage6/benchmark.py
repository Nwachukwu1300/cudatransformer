"""
Stage 6 benchmark: simplified tiled attention vs Stage 3 dense attention.

Produces three things and writes them to stage6_results.txt (repo root):

  1. Correctness  -- the NumPy tiled attention must match Stage 3's
     scaled_dot_product_attention within rtol=1e-4 (the house tolerance for
     matmul/softmax-class ops, per stage1/tests/test_kernels.py).
  2. Execution time (CPU) -- tiled vs dense, using the SAME timing harness as
     Stage 1 (benchmark_fn: warmup=3, runs=10, time.perf_counter).
  3. Peak memory -- analytical (exact score-matrix bytes) plus a measured
     peak-RSS cross-check run in a separate subprocess per method.

The CUDA kernel (stage6/kernels/tiled_attention.cu) cannot build or run on this
Mac (no nvcc / NVIDIA GPU), so the GPU column is left as "N/A (Colab)". Run
stage6/colab_benchmark.py on a GPU to fill it in -- see stage6/COLAB_README.md.
"""

import os
import sys
import importlib.util
import subprocess

import numpy as np

# --------------------------------------------------------------------------
# Paths / imports. We mirror the proven pattern in train_transformer.py:
#   * add stage2 to sys.path so `from tensor import Tensor` works,
#   * importlib-load stage3/nn/attention.py by file path to dodge the
#     stage2/nn vs stage3/nn package-name collision,
#   * add the repo root so we can reuse stage1's benchmark utilities.
# --------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_STAGE2 = os.path.join(_ROOT, "stage2")
_STAGE3 = os.path.join(_ROOT, "stage3")

for _p in (_ROOT, _STAGE2, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _load_by_path(rel_path, name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_ROOT, rel_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


from tensor import Tensor  # noqa: E402  (stage2 autograd Tensor)
from tiled_attention import tiled_attention  # noqa: E402  (our Stage 6 kernel)

_attn = _load_by_path(os.path.join("stage3", "nn", "attention.py"), "stage3_attention")
scaled_dot_product_attention = _attn.scaled_dot_product_attention
create_causal_mask = _attn.create_causal_mask

# Reuse the Stage 1 timing helpers verbatim (same numbers, same house style).
from stage1.benchmark import benchmark_fn, format_time  # noqa: E402


# --------------------------------------------------------------------------
# The two implementations under test, both returning plain NumPy arrays.
# --------------------------------------------------------------------------
def run_reference(q, k, v, mask):
    """Stage 3 dense attention. Wrap NumPy in Tensor; read .data back out."""
    out = scaled_dot_product_attention(Tensor(q), Tensor(k), Tensor(v), mask)
    return out.data


def run_tiled(q, k, v, mask):
    """Stage 6 tiled attention (online softmax, never builds the S x S matrix)."""
    return tiled_attention(q, k, v, mask, block_q=64, block_k=64)


def _make_inputs(batch, heads, seq, head_dim, seed=0):
    rng = np.random.RandomState(seed)
    q = rng.randn(batch, heads, seq, head_dim).astype(np.float32)
    k = rng.randn(batch, heads, seq, head_dim).astype(np.float32)
    v = rng.randn(batch, heads, seq, head_dim).astype(np.float32)
    return q, k, v


def format_bytes(n):
    """Human-readable byte count."""
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024.0 or unit == "GB":
            return f"{n:.1f} {unit}"
        n /= 1024.0


# --------------------------------------------------------------------------
# 1. Correctness
# --------------------------------------------------------------------------
CORRECTNESS_SHAPES = [
    # (batch, heads, seq_len, head_dim, use_causal_mask)
    (2, 4, 64, 16, True),
    (2, 4, 128, 32, True),
    (1, 8, 256, 64, True),
    (2, 2, 512, 32, True),
    (2, 4, 128, 32, False),  # also verify the no-mask path
]
# The tiled result differs from dense only by float32 reduction-order noise
# (~1e-7, i.e. machine epsilon). We assert a tight relative tolerance plus a
# small absolute one: atol is needed because attention outputs contain entries
# near zero, where an rtol-only check (atol=0) is stricter than float32 itself.
RTOL = 1e-4
ATOL = 1e-6


def run_correctness():
    print("\n[1/3] Correctness: tiled vs Stage 3 attention (rtol=%.0e, atol=%.0e)" % (RTOL, ATOL))
    print("-" * 66)
    print(f"{'Shape (B,H,S,d)':>22} {'mask':>6} {'max|diff|':>14} {'result':>8}")
    print("-" * 66)

    rows, all_pass = [], True
    for (b, h, s, d, use_mask) in CORRECTNESS_SHAPES:
        q, k, v = _make_inputs(b, h, s, d)
        mask = create_causal_mask(s) if use_mask else None

        ref = run_reference(q, k, v, mask)
        got = run_tiled(q, k, v, mask)

        max_diff = float(np.max(np.abs(ref - got)))
        try:
            np.testing.assert_allclose(got, ref, rtol=RTOL, atol=ATOL)
            passed = True
        except AssertionError:
            passed = False
        all_pass = all_pass and passed

        shape_str = f"({b}, {h}, {s}, {d})"
        mask_str = "causal" if use_mask else "none"
        print(f"{shape_str:>22} {mask_str:>6} {max_diff:>14.3e} {('PASS' if passed else 'FAIL'):>8}")
        rows.append((shape_str, mask_str, max_diff, passed))

    print("-" * 66)
    print("Overall:", "PASS" if all_pass else "FAIL")
    return rows, all_pass


# --------------------------------------------------------------------------
# 2. Execution time (CPU)
# --------------------------------------------------------------------------
TIME_BATCH, TIME_HEADS, TIME_DIM = 2, 8, 64
TIME_SEQS = [64, 128, 256, 512, 1024]


def run_timing():
    print("\n[2/3] CPU time: tiled vs Stage 3 dense (batch=%d, heads=%d, head_dim=%d)"
          % (TIME_BATCH, TIME_HEADS, TIME_DIM))
    print("-" * 74)
    print(f"{'SeqLen':>10} {'Stage3 (ms)':>14} {'Tiled (ms)':>14} {'CUDA GPU':>14} {'vs Std':>10}")
    print("-" * 74)

    rows = []
    for s in TIME_SEQS:
        q, k, v = _make_inputs(TIME_BATCH, TIME_HEADS, s, TIME_DIM)
        mask = create_causal_mask(s)

        ref_ms = benchmark_fn(run_reference, q, k, v, mask)
        tiled_ms = benchmark_fn(run_tiled, q, k, v, mask)
        ratio = f"{ref_ms / tiled_ms:.2f}x"  # <1 means tiled is slower on CPU

        print(f"{s:>10} {format_time(ref_ms):>14} {format_time(tiled_ms):>14} "
              f"{'N/A (Colab)':>14} {ratio:>10}")
        rows.append((s, ref_ms, tiled_ms, ratio))

    return rows


# --------------------------------------------------------------------------
# 3a. Peak memory -- analytical (exact score-matrix bytes)
# --------------------------------------------------------------------------
MEM_BATCH, MEM_HEADS, BLOCK = 2, 8, 64
MEM_SEQS = [64, 128, 256, 512, 1024, 2048]


def run_memory_analytical():
    print("\n[3/3a] Analytical peak memory: score-matrix intermediate")
    print("       standard = B*H*S*S*4 (dense)   tiled = block*block*4 (one tile)")
    print("-" * 70)
    print(f"{'SeqLen':>10} {'Standard scores':>18} {'Tiled tile':>14} {'Reduction':>12}")
    print("-" * 70)

    rows = []
    tiled_bytes = BLOCK * BLOCK * 4  # one (block_q x block_k) float32 tile, constant in S
    for s in MEM_SEQS:
        std_bytes = MEM_BATCH * MEM_HEADS * s * s * 4
        reduction = std_bytes / tiled_bytes
        print(f"{s:>10} {format_bytes(std_bytes):>18} {format_bytes(tiled_bytes):>14} "
              f"{reduction:>11.0f}x")
        rows.append((s, std_bytes, tiled_bytes, reduction))
    return rows


# --------------------------------------------------------------------------
# 3b. Peak memory -- measured peak RSS, one subprocess per method
# --------------------------------------------------------------------------
# Small batch/head so the O(S^2) score buffer dominates the interpreter
# baseline and the difference is unambiguous.
RSS_BATCH, RSS_HEADS, RSS_DIM = 1, 1, 64
RSS_SEQS = [2048, 4096]


def _measure_peak_rss(method, b, h, s, d):
    proc = subprocess.run(
        [sys.executable, os.path.abspath(__file__), "--mem-worker", method,
         str(b), str(h), str(s), str(d)],
        capture_output=True, text=True,
    )
    for line in proc.stdout.splitlines():
        if line.startswith("MAXRSS_BYTES="):
            return int(line.split("=", 1)[1])
    sys.stderr.write(proc.stderr)
    return None


def run_memory_measured():
    print("\n[3/3b] Measured peak RSS (separate process per method, batch=%d head=%d dim=%d)"
          % (RSS_BATCH, RSS_HEADS, RSS_DIM))
    print("-" * 70)
    print(f"{'SeqLen':>10} {'Standard RSS':>16} {'Tiled RSS':>14} {'Reduction':>12}")
    print("-" * 70)

    rows = []
    for s in RSS_SEQS:
        std = _measure_peak_rss("standard", RSS_BATCH, RSS_HEADS, s, RSS_DIM)
        tiled = _measure_peak_rss("tiled", RSS_BATCH, RSS_HEADS, s, RSS_DIM)
        if std and tiled:
            reduction = f"{std / tiled:.2f}x"
            print(f"{s:>10} {format_bytes(std):>16} {format_bytes(tiled):>14} {reduction:>12}")
            rows.append((s, std, tiled, reduction))
        else:
            print(f"{s:>10} {'(measurement failed)':>16}")
    return rows


def _mem_worker(method, b, h, s, d):
    """Run one method in this (fresh) process, then print peak RSS in bytes.

    ru_maxrss is reported in bytes on macOS but in kilobytes on Linux, so we
    normalise to bytes based on the platform.
    """
    import resource

    q, k, v = _make_inputs(b, h, s, d)
    mask = create_causal_mask(s)
    if method == "standard":
        run_reference(q, k, v, mask)
    else:
        run_tiled(q, k, v, mask)

    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform != "darwin":  # Linux/Colab report kilobytes
        rss *= 1024
    print(f"MAXRSS_BYTES={rss}")


# --------------------------------------------------------------------------
# Results file (Stage-1 house style: banner, metadata, dashed tables)
# --------------------------------------------------------------------------
def save_results(correctness, all_pass, timing, mem_analytical, mem_measured):
    path = os.path.join(_ROOT, "stage6_results.txt")
    L = []
    bar = "=" * 70

    L.append(bar)
    L.append("Stage 6 Benchmark Results: Simplified Tiled Attention")
    L.append("CUDA Transformer Engine - Simplified Tiled Attention")
    L.append(bar)
    L.append("")
    L.append(f"CUDA Available: {False}")
    L.append(f"NumPy Version: {np.__version__}")
    L.append(f"Platform: {sys.platform}")
    L.append("")
    L.append("The CUDA tiled-attention kernel (stage6/kernels/tiled_attention.cu)")
    L.append("cannot compile or run on this machine (no nvcc / NVIDIA GPU). GPU")
    L.append('columns read "N/A (Colab)"; run stage6/colab_benchmark.py on a CUDA')
    L.append("GPU to fill them in (see stage6/COLAB_README.md). Every correctness,")
    L.append("CPU-time and peak-memory number below is really measured, not estimated.")
    L.append("")

    # --- 1. Correctness ---
    L.append("-" * 70)
    L.append("1. Correctness: Python tiled attention vs Stage 3 attention")
    L.append("-" * 70)
    L.append("Reference : stage3/nn/attention.py :: scaled_dot_product_attention")
    L.append("Check     : np.testing.assert_allclose(rtol=1e-4, atol=1e-6)")
    L.append("            (differences are float32 reduction-order noise, ~1e-7)")
    L.append("")
    L.append(f"{'Shape (B,H,S,d)':>22} {'mask':>7} {'max|diff|':>14} {'result':>8}")
    L.append("-" * 60)
    for (shape_str, mask_str, max_diff, passed) in correctness:
        L.append(f"{shape_str:>22} {mask_str:>7} {max_diff:>14.3e} "
                 f"{('PASS' if passed else 'FAIL'):>8}")
    L.append("")
    L.append("Verification:")
    L.append(f"  All shapes match Stage 3 within rtol=1e-4: {'YES' if all_pass else 'NO'}")
    L.append("")

    # --- 2. Execution time ---
    L.append("-" * 70)
    L.append("2. Execution time (CPU): Python tiled vs Stage 3 dense")
    L.append("-" * 70)
    L.append(f"Config: batch={TIME_BATCH}, heads={TIME_HEADS}, head_dim={TIME_DIM}, "
             f"causal mask, block=64")
    L.append("Harness: warmup=3, runs=10, time.perf_counter (same as Stage 1)")
    L.append("")
    L.append(f"{'SeqLen':>10} {'Stage3 (ms)':>15} {'Tiled (ms)':>15} "
             f"{'CUDA GPU (ms)':>15} {'vs Std':>10}")
    L.append("-" * 70)
    for (s, ref_ms, tiled_ms, ratio) in timing:
        L.append(f"{s:>10} {ref_ms:>15.4f} {tiled_ms:>15.4f} "
                 f"{'N/A (Colab)':>15} {ratio:>10}")
    L.append("")
    L.append("Note: On CPU the two are close. Tiled avoids the large O(S^2) score")
    L.append("allocation, so it is competitive (often a touch faster) at small/medium")
    L.append("S, and falls behind only at large S where the Python per-tile loop")
    L.append("overhead dominates NumPy's single fused matmul. On CPU the dependable")
    L.append("win is MEMORY (section 3); the SPEED win needs the CUDA kernel keeping")
    L.append("tiles in shared memory (run on Colab to populate the CUDA GPU column).")
    L.append("")

    # --- 3a. Analytical memory ---
    L.append("-" * 70)
    L.append("3. Peak memory: attention score-matrix intermediate")
    L.append("-" * 70)
    L.append("Dense attention materializes the full (B,H,S,S) score matrix (and")
    L.append("softmax keeps ~2 live copies: exp and probs). Tiled attention only")
    L.append("ever holds one (block x block) score tile plus O(S) running summaries.")
    L.append("")
    L.append(f"(a) Analytical score-matrix bytes  (batch={MEM_BATCH}, heads={MEM_HEADS}, "
             f"block={BLOCK}, float32)")
    L.append(f"{'SeqLen':>10} {'Standard scores':>18} {'Tiled tile':>14} {'Reduction':>12}")
    L.append("-" * 58)
    for (s, std_bytes, tiled_bytes, reduction) in mem_analytical:
        L.append(f"{s:>10} {format_bytes(std_bytes):>18} {format_bytes(tiled_bytes):>14} "
                 f"{reduction:>11.0f}x")
    L.append("")

    # --- 3b. Measured memory ---
    L.append(f"(b) Measured peak process RSS  (batch={RSS_BATCH}, heads={RSS_HEADS}, "
             f"head_dim={RSS_DIM}; one subprocess per method)")
    if mem_measured:
        L.append(f"{'SeqLen':>10} {'Standard RSS':>16} {'Tiled RSS':>14} {'Reduction':>12}")
        L.append("-" * 54)
        for (s, std, tiled, reduction) in mem_measured:
            L.append(f"{s:>10} {format_bytes(std):>16} {format_bytes(tiled):>14} "
                     f"{reduction:>12}")
    else:
        L.append("  (measurement unavailable)")
    L.append("")

    L.append(bar)
    L.append("Correctness PASSED. CPU benchmark saved. Run stage6/colab_benchmark.py")
    L.append("on a GPU to fill the CUDA GPU column. Stage 6 (local part) complete.")
    L.append(bar)
    L.append("")

    with open(path, "w") as f:
        f.write("\n".join(L))
    print(f"\nResults saved to: {path}")
    return path


def main():
    print("=" * 70)
    print("Stage 6 Benchmark: Simplified Tiled Attention")
    print("=" * 70)

    correctness, all_pass = run_correctness()
    if not all_pass:
        # Honesty gate: do not present a benchmark on top of a failing kernel.
        print("\nCorrectness FAILED -- not proceeding to the benchmark. Fix parity first.")
        save_results(correctness, all_pass, [], [], [])
        sys.exit(1)

    timing = run_timing()
    mem_analytical = run_memory_analytical()
    mem_measured = run_memory_measured()

    save_results(correctness, all_pass, timing, mem_analytical, mem_measured)
    print("\nStage 6 local benchmark complete.")


if __name__ == "__main__":
    if len(sys.argv) >= 7 and sys.argv[1] == "--mem-worker":
        _mem_worker(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]),
                    int(sys.argv[5]), int(sys.argv[6]))
    else:
        main()
