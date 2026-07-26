#include <cuda_runtime.h>
#include <stdio.h>

/**
 * Element-wise Operations
 *
 * These kernels implement basic element-wise operations needed for autograd:
 * - multiply: c[i] = a[i] * b[i] (Hadamard product, used in chain rule)
 * - divide: c[i] = a[i] / (b[i] + eps) (used in normalization gradients)
 * - subtract: c[i] = a[i] - b[i] (used in gradient computations)
 *
 * Thread strategy: 256 threads/block, 1D grid (same as vector_add from Stage 1)
 */

// ================================
// Element-wise multiply kernel
// ================================
__global__ void element_wise_multiply_kernel(const float* a, const float* b,
                                             float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] * b[idx];
    }
}

// Host wrapper for element-wise multiply
void element_wise_multiply_cuda(const float* h_a, const float* h_b,
                                float* h_c, int n) {
    float *d_a, *d_b, *d_c;

    // Allocate device memory
    cudaMalloc(&d_a, n * sizeof(float));
    cudaMalloc(&d_b, n * sizeof(float));
    cudaMalloc(&d_c, n * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_a, h_a, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, n * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    int threads_per_block = 256;
    int num_blocks = (n + threads_per_block - 1) / threads_per_block;
    element_wise_multiply_kernel<<<num_blocks, threads_per_block>>>(d_a, d_b, d_c, n);

    // Synchronize
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(h_c, d_c, n * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}

// ================================
// Element-wise divide kernel
// ================================
__global__ void element_wise_divide_kernel(const float* a, const float* b,
                                           float* c, int n, float eps) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Add epsilon for numerical stability
        c[idx] = a[idx] / (b[idx] + eps);
    }
}

// Host wrapper for element-wise divide
void element_wise_divide_cuda(const float* h_a, const float* h_b,
                              float* h_c, int n, float eps) {
    float *d_a, *d_b, *d_c;

    // Allocate device memory
    cudaMalloc(&d_a, n * sizeof(float));
    cudaMalloc(&d_b, n * sizeof(float));
    cudaMalloc(&d_c, n * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_a, h_a, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, n * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    int threads_per_block = 256;
    int num_blocks = (n + threads_per_block - 1) / threads_per_block;
    element_wise_divide_kernel<<<num_blocks, threads_per_block>>>(d_a, d_b, d_c, n, eps);

    // Synchronize
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(h_c, d_c, n * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}

// ================================
// Element-wise subtract kernel
// ================================
__global__ void element_wise_subtract_kernel(const float* a, const float* b,
                                             float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] - b[idx];
    }
}

// Host wrapper for element-wise subtract
void element_wise_subtract_cuda(const float* h_a, const float* h_b,
                                float* h_c, int n) {
    float *d_a, *d_b, *d_c;

    // Allocate device memory
    cudaMalloc(&d_a, n * sizeof(float));
    cudaMalloc(&d_b, n * sizeof(float));
    cudaMalloc(&d_c, n * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_a, h_a, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, n * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    int threads_per_block = 256;
    int num_blocks = (n + threads_per_block - 1) / threads_per_block;
    element_wise_subtract_kernel<<<num_blocks, threads_per_block>>>(d_a, d_b, d_c, n);

    // Synchronize
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(h_c, d_c, n * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}
