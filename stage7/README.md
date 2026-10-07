# Stage 7: TPU Training Comparison

**Goal:** port training to a TPU using JAX and Flax, and benchmark speed,
memory, and cost against the GPU version from earlier stages.

## Why JAX here and nowhere else

CUDA kernels can't run on a TPU. JAX/Flax is the only way to get an identical
training script running on CPU, GPU, *and* TPU with nothing but the hardware
changing — so it's used here, deliberately isolated from the rest of the
project's no-ML-framework constraint:
[`requirements_stage7.txt`](../requirements_stage7.txt) is a separate
dependency file, and nothing outside `stage7/` imports JAX.

## What's here

[`model.py`](model.py) — a Flax reimplementation of the exact Stage 3
architecture (bias-free attention projections, pre-norm blocks, tanh-GELU
FFN, He-init), matched detail-for-detail so the comparison is fair. Asserts
1,311,488 parameters — the same count as Stage 3/4 — before training starts.

[`data.py`](data.py) — loads the real Stage 4 tokenizer and
`create_sequences` by file path, so training data is identical to Stage 4's.

[`train_benchmark.py`](train_benchmark.py) — device-agnostic: the same script
runs unchanged on CPU, GPU, or TPU. Reports time, throughput, peak memory,
estimated cost, and final loss, accumulating one row per device into
`stage7_results.json`/`.txt`.

## Deliverable

```bash
# Local smoke test (CPU, no GPU/TPU needed — confirms correctness)
python3 -m venv .venv_stage7 && source .venv_stage7/bin/activate
pip install jax jaxlib flax optax
python3 stage7/train_benchmark.py --smoke
```

Real GPU and TPU numbers need Colab — full steps, including how to get the
gitignored tokenizer/data onto Colab without uploading the full dataset, in
[COLAB_README.md](COLAB_README.md).

```bash
python3 stage7/train_benchmark.py   # full 15-epoch run, same script on any device
```

## Results

| Device | Time | Steps/s | Peak mem | Est. cost | Train loss | Val loss |
|---|---|---|---|---|---|---|
| Tesla T4 (Colab GPU) | 149.6s | 221.3 | 140.2 MB | $0.0145 | 1.705 | 2.636 |
| **TPU v5e** (Colab TPU) | **58.0s** | **570.8** | 90.8 MB | $0.0193 | 1.705 | 2.638 |
| CPU (Stage 4 NumPy engine, reference) | 124.0 min | — | — | — | 1.693 | 2.656 |

TPU was **2.6x faster than the T4 GPU** and **~128x faster** than the
from-scratch CPU engine, while reaching essentially the Stage 4 target loss —
confirming the reimplementation is faithful to the original architecture, not
just fast. Caveat (also in `stage7_results.txt`): at 1.3M params this shows
mechanics and relative speed, not TPU's real large-scale advantage.

Full numbers: [`stage7_results.txt`](../stage7_results.txt).
All stages: [BENCHMARKS.md](../BENCHMARKS.md).

## A Colab gotcha worth knowing

If a notebook cell calls `jax.devices()` directly (outside the
`!python train_benchmark.py` subprocess), it claims the TPU in the notebook's
own kernel process — a later `!python stage7/train_benchmark.py` subprocess
then fails with `"TPU already in use by process <pid>"`, where that pid is the
notebook's own kernel, not a stray process. Fix: never call `jax.*` in a bare
notebook cell; the script already prints the device line itself
(`JAX x.x.x | backend=... | device=...`).

## Next

Stage 8 (this stage's sibling) adds the documentation you're reading now. See
the root [README.md](../README.md).
