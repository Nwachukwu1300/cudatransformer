#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

/**
 * Mathematical Operations
 *
 * These kernels implement mathematical functions:
 * - exp: c[i] = exp(a[i]) (used in GELU, cross entropy, softmax)
 * - log: c[i] = log(a[i] + eps) (used in cross entropy loss)
 *
 * Thread strategy: 256 threads/block, 1D grid
 */

// ================================
// Exponential kernel
// ================================
__global__ void exp_kernel(const float* a, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = expf(a[idx]);
    }
}

// Host wrapper for exp
void exp_cuda(const float* h_a, float* h_c, int n) {
    float *d_a, *d_c;

    // Allocate device memory
    cudaMalloc(&d_a, n * sizeof(float));
    cudaMalloc(&d_c, n * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_a, h_a, n * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    int threads_per_block = 256;
    int num_blocks = (n + threads_per_block - 1) / threads_per_block;
    exp_kernel<<<num_blocks, threads_per_block>>>(d_a, d_c, n);

    // Synchronize
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(h_c, d_c, n * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_c);
}

// ================================
// Natural logarithm kernel
// ================================
__global__ void log_kernel(const float* a, float* c, int n, float eps) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Add epsilon for numerical stability (avoid log(0))
        c[idx] = logf(a[idx] + eps);
    }
}

// Host wrapper for log
void log_cuda(const float* h_a, float* h_c, int n, float eps) {
    float *d_a, *d_c;

    // Allocate device memory
    cudaMalloc(&d_a, n * sizeof(float));
    cudaMalloc(&d_c, n * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_a, h_a, n * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    int threads_per_block = 256;
    int num_blocks = (n + threads_per_block - 1) / threads_per_block;
    log_kernel<<<num_blocks, threads_per_block>>>(d_a, d_c, n, eps);

    // Synchronize
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(h_c, d_c, n * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_c);
}
