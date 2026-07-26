# Project Context: CUDA Transformer Engine

## What we're building

We're building a transformer from scratch, starting at the CUDA kernel level. We will not use PyTorch or any existing deep learning framework for the core math. We write the GPU kernels, the autograd engine, and the transformer ourselves.

Once the transformer works, we train it on two different tasks using the exact same architecture:

1. A tiny language model that completes sentences (trained on TinyStories or WikiText 2)
2. A sequential recommendation system that predicts the next item a user will pick (trained on MovieLens or similar)

The point of the project is to prove the engine is general. Next token prediction and next item prediction are the same underlying problem: given a sequence, predict what comes next. We use one codebase to solve both.

## Why this project matters

Most portfolio projects call an existing library and fine tune a model. This project shows we understand what happens underneath: memory layout, threading, backward passes, and attention math. The two application demos at the end show the engine generalizes instead of being a one off script.

## Scope for this build

We are building four phases only. Everything beyond that is a stretch goal and should not be started until all four phases work end to end.

1. CUDA kernels: vector add, matrix multiply, softmax, a reduction op
2. A small autograd engine (Tensor class with forward and backward) trained on MNIST using our own kernels
3. A transformer (embeddings, multi head attention, feed forward, layer norm, residuals) trained on TinyStories, producing coherent sentence completions
4. The same transformer retrained on a recommendation dataset, predicting the next item in a user's sequence

Stretch goals we are explicitly deferring: a full FlashAttention implementation, TPU training with JAX and Flax, and heavy documentation and diagrams. We can revisit these after phase 4 is working.

## Technical constraints

- Language: Python for orchestration, CUDA (via a C++ extension, or raw CUDA C with Python bindings) for the kernels
- No PyTorch, TensorFlow, or JAX for the core math. NumPy is fine for CPU comparisons and data loading.
- Each phase needs a working deliverable before we move to the next phase. We do not move forward on a phase that doesn't run.
- We benchmark GPU kernels against a naive CPU version at each stage, so we can show real numbers later.

## How to work with me on this

- Build one phase at a time. Confirm the deliverable works before touching the next phase.
- Keep the code readable over clever. This project needs to be explainable in an interview, not just fast.
- Flag anything that looks like scope creep back to me. If a task starts pulling in a stretch goal, stop and ask first.
