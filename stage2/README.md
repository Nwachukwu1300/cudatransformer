# Stage 2: A Small Deep Learning Engine

**Goal:** build the core of a framework — an autograd engine with the layers
and optimizers needed to train a real model — using the kernel ideas from
Stage 1.

## What's here

[`tensor.py`](tensor.py) — a `Tensor` class that tracks every operation in a
computation graph and supports `.backward()`, the core of the engine. Built in
the same spirit as PyTorch's autograd: a dynamic graph, topological sort for
the backward pass, and gradient accumulation.

`nn/` — the layers built on top of `Tensor`:
- [`linear.py`](nn/linear.py) — fully connected layer
- [`normalization.py`](nn/normalization.py) — LayerNorm
- [`activations.py`](nn/activations.py) — ReLU, GELU
- [`loss.py`](nn/loss.py) — cross-entropy loss
- [`module.py`](nn/module.py) — base class tying layers together

`optim/` — optimizers:
- [`sgd.py`](optim/sgd.py)
- [`adam.py`](optim/adam.py)

`kernels/` — CUDA kernels backing the same operations (activations, cross
entropy, element-wise ops, reductions, optimizer updates, transpose), written
to the same parallelization patterns as Stage 1. See
[CUDA_OPTIMIZATIONS.md](../CUDA_OPTIMIZATIONS.md).

[`models/mlp.py`](models/mlp.py) — the MNIST model: a 3-layer MLP
(784 → 256 → 128 → 10).

**Important:** the live training path (`Tensor` in `tensor.py`) is pure NumPy.
The CUDA kernels in `kernels/*.cu` are a parallel, benchmarked implementation
of the same math, not wired into the training loop — this is why every model
in Stages 2–5 trains on CPU, and is also why this project runs end to end on a
GPU-less machine. See the root [README.md](../README.md#engine-internals) for
the full architecture diagram.

## Deliverable

Train a small MLP on MNIST using only the Stage 1/2 kernels and classes.

```bash
python3 train_mnist.py
```

## Results

| Run | Best test accuracy | Training time |
|---|---|---|
| CPU (local) | **98.00%** | 27.6s |
| GPU (Colab) | 98.14% | 60.1s |

Full training curves: [`stage2_results.txt`](../stage2_results.txt) (CPU),
[`stage2_results_gpu.txt`](../stage2_results_gpu.txt) (GPU), and the detailed
first-person implementation notes in [`STAGE2_REPORT.txt`](../STAGE2_REPORT.txt).
A training curve plot: [`stage2_results_visualization.png`](../stage2_results_visualization.png).

Full numbers for every stage: [BENCHMARKS.md](../BENCHMARKS.md).

## Next

Stage 3 replaces this MLP with a transformer, reusing `Tensor`, `Linear`,
`LayerNorm`, and `Adam` from this stage unchanged. See
[stage3/README.md](../stage3/README.md).
