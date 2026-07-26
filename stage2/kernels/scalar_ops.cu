#include <cuda_runtime.h>
#include <stdio.h>

/**
 * Scalar Operations
 *
 * These kernels implement operations involving scalars:
 * - scalar_multiply: c[i] = scalar * a[i] (used in optimizer updates, learning rate scaling)
 *
 * Thread strategy: 256 threads/block, 1D grid
 */

// ================================
// Scalar multiply kernel
// ================================
__global__ void scalar_multiply_kernel(const float* a, float scalar,
                                       float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = scalar * a[idx];
    }
}

// Host wrapper for scalar multiply
void scalar_multiply_cuda(const float* h_a, float scalar,
                          float* h_c, int n) {
    float *d_a, *d_c;

    // Allocate device memory
    cudaMalloc(&d_a, n * sizeof(float));
    cudaMalloc(&d_c, n * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_a, h_a, n * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    int threads_per_block = 256;
    int num_blocks = (n + threads_per_block - 1) / threads_per_block;
    scalar_multiply_kernel<<<num_blocks, threads_per_block>>>(d_a, scalar, d_c, n);

    // Synchronize
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(h_c, d_c, n * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_c);
}
