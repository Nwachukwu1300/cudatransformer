/*
 * Simplified Tiled Attention CUDA Kernel  (Stage 6)
 * =================================================
 *
 * Computes scaled dot-product attention
 *
 *     O = softmax( (Q @ K^T) * scale  + causal_mask ) @ V
 *
 * for inputs shaped (batch*heads, seq, head_dim), the SAME result as the Stage 3
 * NumPy attention -- but it never writes the (seq x seq) score matrix to global
 * memory. This is an original implementation of the tiling + online-softmax idea
 * (FlashAttention in spirit), written from scratch for this project.
 *
 * ------------------------------------------------------------------------
 * WHY TILING (and where the memory win comes from)
 * ------------------------------------------------------------------------
 * A standard GPU attention built from our Stage 1 kernels would do three passes,
 * each round-tripping through GLOBAL memory:
 *     1) matmul  -> write scores (seq x seq) to global memory
 *     2) softmax -> read scores, write probs (seq x seq) to global memory
 *     3) matmul  -> read probs, multiply by V, write output
 * The (seq x seq) score/probability buffers are O(seq^2) global memory AND the
 * dominant cost is shuffling them in and out of DRAM.
 *
 * This kernel fuses all three steps. Each thread block owns one tile of query
 * rows and streams over the K/V sequence in blocks. A K/V block is loaded ONCE
 * into fast on-chip __shared__ memory and reused by every query row in the block,
 * and the running attention result is kept in registers. The score numbers are
 * produced, consumed, and discarded on-chip; they never touch global memory. So
 * global memory traffic drops to just reading Q,K,V once and writing O once, and
 * the O(seq^2) score buffer is never allocated at all.
 *
 * ------------------------------------------------------------------------
 * THE ONLINE SOFTMAX TRICK (how we softmax a row we only see in pieces)
 * ------------------------------------------------------------------------
 * Softmax needs the row max and the row sum, which are reductions over the whole
 * row. Because we only ever hold one K block at a time, we keep three running
 * numbers per query row and update them as each key is folded in:
 *     m = running max of the scores seen so far
 *     l = running sum of exp(score - m)
 *     acc[0..head_dim) = running sum of exp(score - m) * V_row
 * When a new key gives a score s that raises the max to m_new = max(m, s), every
 * exponential we already accumulated was shifted by the OLD max, so we rescale
 * the old l and acc by the correction factor exp(m_old - m_new) before adding the
 * new key's contribution. After the last key, acc / l equals the exact softmax
 * output for that row. Causal / out-of-range keys are simply skipped (weight 0),
 * which matches the additive -1e9 mask used in Stage 3 to floating-point noise.
 */

#include <cuda_runtime.h>
#include <math.h>

// One thread block processes BLOCK_Q query rows; one thread == one query row.
// The K/V sequence is streamed in blocks of BLOCK_K rows held in shared memory.
#define BLOCK_Q 64
#define BLOCK_K 64
#define MAX_HEAD_DIM 128   // per-thread register arrays are sized for this cap
#define NEG_INF (-1e30f)

__global__ void tiled_attention_kernel(
        const float* __restrict__ Q,   // (num_bh, seq, head_dim)
        const float* __restrict__ K,   // (num_bh, seq, head_dim)
        const float* __restrict__ V,   // (num_bh, seq, head_dim)
        float* __restrict__ O,         // (num_bh, seq, head_dim)
        int num_bh, int seq, int head_dim, float scale, int causal) {

    // Dynamic shared memory holds one K tile and one V tile:
    //   Ks[BLOCK_K * head_dim] followed by Vs[BLOCK_K * head_dim].
    extern __shared__ float smem[];
    float* Ks = smem;
    float* Vs = smem + BLOCK_K * head_dim;

    const int bh   = blockIdx.y;                         // which (batch, head)
    const int row  = blockIdx.x * BLOCK_Q + threadIdx.x; // global query index
    const int tid  = threadIdx.x;

    // Base pointers for this (batch, head) slice.
    const float* Qb = Q + (long)bh * seq * head_dim;
    const float* Kb = K + (long)bh * seq * head_dim;
    const float* Vb = V + (long)bh * seq * head_dim;
    float*       Ob = O + (long)bh * seq * head_dim;

    // Per-thread (per-query-row) state.
    float q_reg[MAX_HEAD_DIM];               // this row's query vector
    float acc[MAX_HEAD_DIM];                 // running output accumulator
    float m_run = NEG_INF;                   // running max
    float l_run = 0.0f;                      // running normaliser

    const bool active = (row < seq);
    if (active) {
        for (int d = 0; d < head_dim; d++) {
            q_reg[d] = Qb[row * head_dim + d];
            acc[d]   = 0.0f;
        }
    }

    // For a causal mask, the largest key any row in this query tile may attend to
    // is the tile's last row, so we can stop the OUTER loop there. The bound is
    // the same for every thread in the block (it does not depend on `row`), which
    // is essential: all threads must execute the same number of __syncthreads().
    int key_limit = seq;
    if (causal) {
        int tile_last_row = (blockIdx.x + 1) * BLOCK_Q;   // exclusive
        key_limit = min(tile_last_row, seq);
    }

    // ----- stream over blocks of keys/values -----
    for (int k0 = 0; k0 < key_limit; k0 += BLOCK_K) {

        // Cooperatively load this K/V block into shared memory. All BLOCK_Q
        // threads help, striding over BLOCK_K*head_dim elements. Rows past the
        // end of the sequence are zero-filled (and later skipped by the mask).
        for (int idx = tid; idx < BLOCK_K * head_dim; idx += blockDim.x) {
            int krow = k0 + idx / head_dim;   // global key index
            int kcol = idx % head_dim;        // dimension within the key vector
            if (krow < seq) {
                Ks[idx] = Kb[krow * head_dim + kcol];
                Vs[idx] = Vb[krow * head_dim + kcol];
            } else {
                Ks[idx] = 0.0f;
                Vs[idx] = 0.0f;
            }
        }
        __syncthreads();   // shared K/V block is now visible to every thread

        // Fold each key in this block into this row's running softmax.
        if (active) {
            for (int j = 0; j < BLOCK_K; j++) {
                int krow = k0 + j;
                if (krow >= seq) break;                 // past sequence end
                if (causal && krow > row) break;        // future key: skip (keys ascend)

                // score = scale * (q . k). Dot product reads K from shared memory.
                float s = 0.0f;
                const float* kj = &Ks[j * head_dim];
                for (int d = 0; d < head_dim; d++) {
                    s += q_reg[d] * kj[d];
                }
                s *= scale;

                // ----- online softmax update for this single key -----
                float m_new = fmaxf(m_run, s);
                float corr  = __expf(m_run - m_new);    // rescales prior mass (=1 when s<=m)
                float p     = __expf(s - m_new);        // this key's un-normalised weight

                l_run = l_run * corr + p;
                const float* vj = &Vs[j * head_dim];
                for (int d = 0; d < head_dim; d++) {
                    acc[d] = acc[d] * corr + p * vj[d];
                }
                m_run = m_new;
            }
        }
        __syncthreads();   // finish reading shared before the next block overwrites it
    }

    // Normalise once and write this row's output. Every causal row attends to at
    // least its own position, so l_run > 0.
    if (active) {
        float inv_l = 1.0f / l_run;
        for (int d = 0; d < head_dim; d++) {
            Ob[row * head_dim + d] = acc[d] * inv_l;
        }
    }
}

/*
 * Host launcher. Mirrors the Stage 1 kernel convention: allocate device buffers,
 * copy Q/K/V up, launch, synchronise, copy O back, free. Inputs are the flat
 * (num_bh, seq, head_dim) buffers of contiguous float32 arrays; `scale` is
 * computed here as 1/sqrt(head_dim) exactly like Stage 3.
 */
void tiled_attention_cuda(const float* hQ, const float* hK, const float* hV,
                          float* hO, int num_bh, int seq, int head_dim, int causal) {
    float *dQ, *dK, *dV, *dO;
    size_t bytes = (size_t)num_bh * seq * head_dim * sizeof(float);

    cudaMalloc(&dQ, bytes);
    cudaMalloc(&dK, bytes);
    cudaMalloc(&dV, bytes);
    cudaMalloc(&dO, bytes);

    cudaMemcpy(dQ, hQ, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dK, hK, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dV, hV, bytes, cudaMemcpyHostToDevice);

    float scale = 1.0f / sqrtf((float)head_dim);

    dim3 block(BLOCK_Q);
    dim3 grid((seq + BLOCK_Q - 1) / BLOCK_Q, num_bh);
    // Shared memory: one K tile + one V tile of the current block.
    size_t shmem = (size_t)2 * BLOCK_K * head_dim * sizeof(float);

    tiled_attention_kernel<<<grid, block, shmem>>>(
        dQ, dK, dV, dO, num_bh, seq, head_dim, scale, causal);

    cudaDeviceSynchronize();

    cudaMemcpy(hO, dO, bytes, cudaMemcpyDeviceToHost);

    cudaFree(dQ);
    cudaFree(dK);
    cudaFree(dV);
    cudaFree(dO);
}
