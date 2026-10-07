# CUDA Optimizations

A walkthrough of the actual optimization techniques used across the kernels in
this repo, with references to the real source. Every technique here is
standard GPU-programming practice; what makes it concrete is that each one is
applied, measured, and compared against a naive baseline in this codebase.

## 1. One thread per element (the baseline pattern)

[`stage1/kernels/vector_add.cu`](stage1/kernels/vector_add.cu) is the simplest
kernel in the repo and sets the pattern every other kernel builds on:

```cuda
int idx = blockIdx.x * blockDim.x + threadIdx.x;
if (idx < n) {
    c[idx] = a[idx] + b[idx];
}
```

- **Grid-stride indexing**: `blockIdx.x * blockDim.x + threadIdx.x` maps each
  thread to exactly one global index — the fundamental CUDA addressing pattern.
- **Bounds check**: `if (idx < n)` is necessary because the grid size is
  rounded up to a multiple of the block size (`(n + threads - 1) / threads`
  blocks), so the last block has threads with no valid element to process.
- **256 threads per block**: a conventional choice that keeps occupancy high
  on most architectures without over-subscribing shared memory or registers.

Every other 1D kernel in the repo (the element-wise ops, scalar multiply, math
ops in Stage 2) follows this exact same shape.

## 2. Shared-memory tiling (matrix multiply)

[`stage1/kernels/matmul.cu`](stage1/kernels/matmul.cu) implements **two**
versions of matmul side by side so the optimization's effect is directly
visible: a naive kernel and a tiled kernel, switchable via a flag
(`matmul_cuda(..., use_tiled)`).

**The problem with the naive version**: each thread computing `C[row, col]`
reads an entire row of `A` and an entire column of `B` from global memory —
`K` reads of `A` per thread, `K` reads of `B` per thread, with massive
redundancy since neighboring threads in the same block re-read nearly the same
data from global memory independently.

**The fix**: load a `BLOCK_SIZE × BLOCK_SIZE` (16×16) tile of `A` and `B` into
`__shared__` memory once per block, synchronize, then have every thread in the
block compute its partial dot product from the on-chip copy:

```cuda
__shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
__shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
...
for (int t = 0; t < (K + BLOCK_SIZE - 1) / BLOCK_SIZE; t++) {
    As[threadIdx.y][threadIdx.x] = A[row * K + a_col];   // one global read per thread
    Bs[threadIdx.y][threadIdx.x] = B[b_row * N + col];
    __syncthreads();                                      // wait for the whole tile to land

    for (int k = 0; k < BLOCK_SIZE; k++)
        sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];   // reused BLOCK_SIZE times, on-chip

    __syncthreads();                                      // wait before overwriting the tile
}
```

Each element loaded from global memory is now reused `BLOCK_SIZE` (16) times
from shared memory instead of being re-fetched from DRAM by every thread that
needs it — shared memory is roughly two orders of magnitude faster than global
memory on typical GPUs, so this turns a memory-bandwidth-bound kernel into a
much better balance of compute and memory traffic. The two `__syncthreads()`
calls are required correctness barriers: the first ensures the whole tile is
loaded before any thread reads from it, the second ensures no thread starts
overwriting the tile for the next iteration before every thread has finished
reading the current one.

## 3. Parallel reduction with warp shuffles (sum reduction, softmax)

[`stage1/kernels/reduce_sum.cu`](stage1/kernels/reduce_sum.cu) and the
reduction helpers inside
[`stage1/kernels/softmax.cu`](stage1/kernels/softmax.cu) both need to combine
many values down to one (a sum, or a max) as fast as possible. Three layered
techniques are used together:

**a) Tree reduction in shared memory**, halving the active thread count each
step instead of summing serially:
```cuda
if (BLOCK_SIZE >= 512) { if (tid < 256) shared[tid] += shared[tid + 256]; __syncthreads(); }
if (BLOCK_SIZE >= 256) { if (tid < 128) shared[tid] += shared[tid + 128]; __syncthreads(); }
if (BLOCK_SIZE >= 128) { if (tid <  64) shared[tid] += shared[tid +  64]; __syncthreads(); }
```
This is O(log n) steps instead of O(n), and the loop is manually unrolled
(compile-time `if (BLOCK_SIZE >= ...)` branches) so the compiler can generate
straight-line code with no runtime loop overhead.

**b) Sequential addressing** — note the reduction above folds `shared[tid]`
with `shared[tid + 256]`, not adjacent threads. This specific pattern avoids
**shared memory bank conflicts**: shared memory is divided into banks, and if
multiple threads in a warp access the same bank simultaneously, those accesses
serialize. Sequential addressing (as opposed to the naive "every other thread"
interleaved pattern) keeps each step's active threads hitting distinct banks.

**c) Warp-level shuffle for the final 32 elements**, which is the fastest
reduction primitive available — no shared memory or `__syncthreads()` needed
once the active set fits in a single warp:
```cuda
__device__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}
```
`__shfl_down_sync` lets threads within a warp exchange register values
directly, without round-tripping through memory at all — the last mile of the
reduction is essentially free compared to a memory-based approach.

The softmax kernel reuses this exact same `warp_reduce_max` /
`block_reduce_max` / `warp_reduce_sum` / `block_reduce_sum` machinery twice per
row (once for the numerically-stable max subtraction, once for the
normalization sum) — a direct example of the reduction primitive being a
reusable building block across kernels.

## 4. One block per row (softmax, with the numerical-stability trick)

[`stage1/kernels/softmax.cu`](stage1/kernels/softmax.cu) assigns one thread
block to each row of the input, so an entire row's reduction (max, then sum)
happens with fast intra-block shared memory instead of needing cross-block
communication:

```cuda
int row = blockIdx.x;
const float* row_input = input + row * num_cols;
```

The kernel does the standard **max-subtraction trick** for numerical stability
— computing `exp(x - max(x))` instead of `exp(x)` directly, since raw
exponentials of typical logit magnitudes can overflow `float32`:

```cuda
float max_val = block_reduce_max(...);      // pass 1: row max
...
float exp_val = expf(row_input[i] - max_val);  // pass 2: shifted exponentials
sum = block_reduce_sum(...);                    // running sum of them
...
row_output[i] = row_output[i] / sum;            // pass 3: normalize
```

Three passes over the row, each parallelized across the block's threads with a
strided loop (`for (int i = threadIdx.x; i < num_cols; i += blockDim.x)`) so
the kernel handles rows longer than the block's thread count correctly.

## 5. Coalesced transpose with shared memory (avoiding stride-N writes)

[`stage2/kernels/transpose.cu`](stage2/kernels/transpose.cu) (used in the
autograd engine's backward pass for `A.T @ grad_output`) is a case where the
*naive* approach has a specific, well-known memory access problem: a direct
`output[j*rows + i] = input[i*cols + j]` transpose has threads in a warp
writing to `output` with stride `rows` apart — the opposite of the
coalesced (contiguous) access pattern the memory controller is built for. The
fix is the same shared-memory staging trick as matmul: load a tile into shared
memory with coalesced reads, then write it back out with coalesced writes,
letting the transpose happen entirely inside the fast on-chip tile instead of
in the global-memory access pattern itself.

## 6. Fusing an entire operator into one kernel pass (Stage 6 tiled attention)

[`stage6/kernels/tiled_attention.cu`](stage6/kernels/tiled_attention.cu) is the
most involved kernel in the repo and combines several of the techniques above
with one additional idea: **fusion + online softmax**, in the spirit of
FlashAttention.

**The problem it solves**: a standard attention implementation built from the
Stage 1 primitives would be three separate kernel launches, each round-tripping
through global memory —
```
matmul (Q·Kᵀ) -> write (seq×seq) scores to DRAM
softmax        -> read scores, write (seq×seq) probs to DRAM
matmul (·V)    -> read probs, write output to DRAM
```
For `seq=1024`, that score/probability buffer alone is **64 MB** per
`(batch, head)` slice (measured in `stage6_results_gpu.txt`) — and it's pure
memory traffic, not useful compute.

**The fix**: fuse all three steps into one kernel. Each thread block owns a
tile of query rows and streams the K/V sequence through shared memory in
blocks (`BLOCK_K = 64` rows at a time, matching the tiling pattern from matmul
— load once into `__shared__`, reuse across every query row in the block):

```cuda
#define BLOCK_Q 64
#define BLOCK_K 64
extern __shared__ float smem[];
float* Ks = smem;
float* Vs = smem + BLOCK_K * head_dim;
```

The catch is softmax needs the max and sum over the *entire* row, but the
kernel only ever holds one K/V block in shared memory at a time — it never
sees the whole row at once. The **online softmax** trick solves this by
keeping three running values per query row and updating them incrementally as
each K block streams through:

```
m = running max of scores seen so far
l = running sum of exp(score - m)
acc = running sum of exp(score - m) * V_row
```

When a new block raises the max from `m` to `m_new`, the previously
accumulated `l` and `acc` were scaled relative to the *old* max — so they're
corrected by a rescaling factor `exp(m_old - m_new)` before the new block's
contribution is folded in. After the last K block, `acc / l` is the exact
softmax-weighted output for that row — mathematically identical to computing
the full row at once, just computed incrementally. This is what makes it
possible to never materialize the `(seq × seq)` matrix at all: the score
values for a given K block are produced, consumed into the running
accumulators, and discarded before the next block is even loaded.

**Measured effect** (`stage6_results_gpu.txt`, Tesla T4): 2.5x faster at
seq=64, climbing to **25.7x faster** at seq=1024 as the avoided O(seq²) memory
traffic dominates more and more of the naive version's cost — and **128 MB of
global memory never allocated** at seq=1024, since the tiled kernel only ever
holds `Q, K, V, O` (16 MB total) plus one `BLOCK_K × head_dim` tile in shared
memory, regardless of sequence length.

## Summary table

| Technique | Where | What it avoids |
|---|---|---|
| One-thread-per-element + bounds check | `vector_add.cu`, most Stage 2 kernels | Baseline parallel pattern |
| Shared-memory tiling | `matmul.cu`, `transpose.cu` | Redundant global memory reads across threads in a block |
| Tree reduction + sequential addressing | `reduce_sum.cu` | O(n) serial summation; shared-memory bank conflicts |
| Warp-shuffle reduction | `reduce_sum.cu`, `softmax.cu` | Shared memory / `__syncthreads()` overhead for the last 32 elements |
| One block per row + max-subtraction | `softmax.cu` | Numerical overflow in `exp()`; cross-block synchronization |
| Coalesced tile staging | `transpose.cu` | Strided (non-coalesced) global memory writes |
| Kernel fusion + online softmax | `stage6/kernels/tiled_attention.cu` | The entire O(seq²) score/probability buffer in global memory |

Each of these is a small idea on its own; what Stage 6 shows is that they
compose — the tiled attention kernel uses shared-memory tiling (from matmul),
per-row block ownership (from softmax), and adds the online-softmax
accumulator trick on top, to fuse what would otherwise be three
memory-bound kernel launches into one.
