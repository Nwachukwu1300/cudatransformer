/*
 * Softmax CUDA Kernel
 *
 * Computes softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
 * along the last axis of a 2D array.
 *
 * Parallelization strategy:
 * - One block per row
 * - Threads within a block cooperate to:
 *   1. Find the max value in the row (parallel reduction)
 *   2. Compute exp(x - max) for each element
 *   3. Sum the exponentials (parallel reduction)
 *   4. Divide each element by the sum
 *
 * Uses shared memory for efficient reductions within a block.
 */

#include <cuda_runtime.h>
#include <float.h>

#define BLOCK_SIZE 256

// Warp-level reduction for max
__device__ float warp_reduce_max(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

// Warp-level reduction for sum
__device__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Block-level reduction for max using shared memory
__device__ float block_reduce_max(float val, float* shared) {
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    // Warp-level reduction
    val = warp_reduce_max(val);

    // Write warp results to shared memory
    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();

    // First warp reduces across warps
    val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : -FLT_MAX;
    if (wid == 0) {
        val = warp_reduce_max(val);
    }

    return val;
}

// Block-level reduction for sum using shared memory
__device__ float block_reduce_sum(float val, float* shared) {
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    val = warp_reduce_sum(val);

    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();

    val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : 0.0f;
    if (wid == 0) {
        val = warp_reduce_sum(val);
    }

    return val;
}

// Softmax kernel: one block per row
__global__ void softmax_kernel(const float* input, float* output,
                               int num_rows, int num_cols) {
    __shared__ float shared[32];  // For warp reduction results

    int row = blockIdx.x;
    if (row >= num_rows) return;

    const float* row_input = input + row * num_cols;
    float* row_output = output + row * num_cols;

    // Step 1: Find max value in this row
    float max_val = -FLT_MAX;
    for (int i = threadIdx.x; i < num_cols; i += blockDim.x) {
        max_val = fmaxf(max_val, row_input[i]);
    }
    max_val = block_reduce_max(max_val, shared);

    // Broadcast max to all threads
    __shared__ float row_max;
    if (threadIdx.x == 0) {
        row_max = max_val;
    }
    __syncthreads();
    max_val = row_max;

    // Step 2: Compute exp(x - max) and sum
    float sum = 0.0f;
    for (int i = threadIdx.x; i < num_cols; i += blockDim.x) {
        float exp_val = expf(row_input[i] - max_val);
        row_output[i] = exp_val;  // Temporarily store exp values
        sum += exp_val;
    }
    sum = block_reduce_sum(sum, shared);

    // Broadcast sum to all threads
    __shared__ float row_sum;
    if (threadIdx.x == 0) {
        row_sum = sum;
    }
    __syncthreads();
    sum = row_sum;

    // Step 3: Normalize by sum
    for (int i = threadIdx.x; i < num_cols; i += blockDim.x) {
        row_output[i] = row_output[i] / sum;
    }
}

// Host function to launch the kernel
void softmax_cuda(const float* h_input, float* h_output,
                  int num_rows, int num_cols) {
    float *d_input, *d_output;
    size_t size = num_rows * num_cols * sizeof(float);

    // Allocate device memory
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);

    // Copy input to device
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

    // Launch kernel: one block per row
    int threads = min(BLOCK_SIZE, num_cols);
    // Round up to nearest power of 2 for efficient reductions
    threads = 1 << (int)ceil(log2f((float)threads));
    threads = max(32, min(threads, BLOCK_SIZE));

    softmax_kernel<<<num_rows, threads>>>(d_input, d_output, num_rows, num_cols);

    cudaDeviceSynchronize();

    // Copy result back
    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
}
