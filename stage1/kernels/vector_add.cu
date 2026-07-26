/*
 * Vector Addition CUDA Kernel
 *
 * Parallelization strategy:
 * - One thread per element
 * - 1D grid of 1D blocks
 * - Each thread computes c[i] = a[i] + b[i]
 *
 * This is the simplest possible kernel - a good starting point for
 * understanding CUDA threading and memory access patterns.
 */

#include <cuda_runtime.h>

// Kernel: Each thread adds one element
__global__ void vector_add_kernel(const float* a, const float* b, float* c, int n) {
    // Global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Bounds check - important when n is not a multiple of block size
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// Host function to launch the kernel
void vector_add_cuda(const float* h_a, const float* h_b, float* h_c, int n) {
    float *d_a, *d_b, *d_c;
    size_t size = n * sizeof(float);

    // Allocate device memory
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    // Copy inputs to device
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    // Launch kernel
    // 256 threads per block is a common choice for good occupancy
    int threads_per_block = 256;
    int num_blocks = (n + threads_per_block - 1) / threads_per_block;

    vector_add_kernel<<<num_blocks, threads_per_block>>>(d_a, d_b, d_c, n);

    // Wait for kernel to finish
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}
