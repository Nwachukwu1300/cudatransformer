/*
 * Sum Reduction CUDA Kernel
 *
 * Computes the sum of all elements in an array using parallel reduction.
 *
 * Parallelization strategy:
 * - Tree-based reduction within each block using shared memory
 * - Multiple blocks handle large arrays
 * - Final reduction across block results
 *
 * Key optimizations demonstrated:
 * - Shared memory for fast intra-block communication
 * - Sequential addressing to avoid bank conflicts
 * - Loop unrolling for the last warp
 * - Warp shuffle for final reduction (fastest on modern GPUs)
 */

#include <cuda_runtime.h>

#define BLOCK_SIZE 256

// Warp-level sum reduction using shuffle
__device__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// First-level reduction: reduces input array to partial sums
__global__ void reduce_sum_kernel(const float* input, float* output, int n) {
    __shared__ float shared[BLOCK_SIZE];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    // Load two elements per thread and add them
    float sum = 0.0f;
    if (idx < n) {
        sum = input[idx];
    }
    if (idx + blockDim.x < n) {
        sum += input[idx + blockDim.x];
    }

    shared[tid] = sum;
    __syncthreads();

    // Tree reduction in shared memory
    // Unroll the loop for better performance
    if (BLOCK_SIZE >= 512) {
        if (tid < 256) { shared[tid] += shared[tid + 256]; }
        __syncthreads();
    }
    if (BLOCK_SIZE >= 256) {
        if (tid < 128) { shared[tid] += shared[tid + 128]; }
        __syncthreads();
    }
    if (BLOCK_SIZE >= 128) {
        if (tid < 64) { shared[tid] += shared[tid + 64]; }
        __syncthreads();
    }

    // Final warp reduction (no sync needed within a warp)
    if (tid < 32) {
        float val = shared[tid];
        if (BLOCK_SIZE >= 64) val += shared[tid + 32];
        val = warp_reduce_sum(val);

        if (tid == 0) {
            output[blockIdx.x] = val;
        }
    }
}

// Single-block final reduction for small arrays
__global__ void reduce_sum_final_kernel(const float* input, float* output, int n) {
    __shared__ float shared[BLOCK_SIZE];

    int tid = threadIdx.x;
    float sum = 0.0f;

    // Each thread sums multiple elements
    for (int i = tid; i < n; i += blockDim.x) {
        sum += input[i];
    }

    shared[tid] = sum;
    __syncthreads();

    // Tree reduction
    for (int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            shared[tid] += shared[tid + s];
        }
        __syncthreads();
    }

    // Final warp reduction
    if (tid < 32) {
        float val = shared[tid];
        if (blockDim.x >= 64) val += shared[tid + 32];
        val = warp_reduce_sum(val);

        if (tid == 0) {
            output[0] = val;
        }
    }
}

// Host function to launch the kernel
float reduce_sum_cuda(const float* h_input, int n) {
    float *d_input, *d_partial, *d_result;
    float result;

    // Allocate device memory
    cudaMalloc(&d_input, n * sizeof(float));
    cudaMemcpy(d_input, h_input, n * sizeof(float), cudaMemcpyHostToDevice);

    // Calculate grid size
    // Each block processes 2 * BLOCK_SIZE elements
    int num_blocks = (n + 2 * BLOCK_SIZE - 1) / (2 * BLOCK_SIZE);

    if (num_blocks == 1) {
        // Small array: single kernel call
        cudaMalloc(&d_result, sizeof(float));
        reduce_sum_final_kernel<<<1, BLOCK_SIZE>>>(d_input, d_result, n);
        cudaMemcpy(&result, d_result, sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(d_result);
    } else {
        // Large array: two-level reduction
        cudaMalloc(&d_partial, num_blocks * sizeof(float));
        cudaMalloc(&d_result, sizeof(float));

        // First level: reduce to partial sums
        reduce_sum_kernel<<<num_blocks, BLOCK_SIZE>>>(d_input, d_partial, n);

        // Second level: reduce partial sums
        reduce_sum_final_kernel<<<1, BLOCK_SIZE>>>(d_partial, d_result, num_blocks);

        cudaMemcpy(&result, d_result, sizeof(float), cudaMemcpyDeviceToHost);

        cudaFree(d_partial);
        cudaFree(d_result);
    }

    cudaFree(d_input);

    return result;
}

// Version that writes to an output array (for consistency with other kernels)
void reduce_sum_cuda_array(const float* h_input, float* h_output, int n) {
    h_output[0] = reduce_sum_cuda(h_input, n);
}
