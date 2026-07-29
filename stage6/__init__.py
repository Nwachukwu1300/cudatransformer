"""
Stage 6: Simplified Tiled Attention.

An original, from-first-principles implementation of block-wise ("tiled")
attention with the online-softmax trick. The goal is the same idea that makes
FlashAttention fast -- never build the full (seq x seq) attention matrix -- but
written from scratch here, not copied from any existing implementation.

Contents:
- tiled_attention.py : plain NumPy reference for the tiled algorithm (CPU, runs anywhere)
- kernels/tiled_attention.cu + cuda_ext.cpp : the same logic as a CUDA kernel (GPU/Colab)
- benchmark.py : correctness check vs Stage 3 attention + CPU time/peak-memory benchmark
- colab_benchmark.py : turnkey script to build + benchmark the CUDA kernel on a GPU
"""
