#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

/**
 * Optimizer Update Kernels
 *
 * Implements parameter update rules for optimizers:
 * - SGD: param -= lr * grad
 * - Adam: param -= lr * m_hat / (sqrt(v_hat) + eps)
 *         with momentum (m) and variance (v) tracking
 *
 * Thread strategy: 256 threads/block, 1D grid
 */

// ================================
// SGD Update
// ================================
__global__ void sgd_update_kernel(float* params, const float* grads,
                                 float learning_rate, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        params[idx] -= learning_rate * grads[idx];
    }
}

void sgd_update_cuda(float* h_params, const float* h_grads,
                    float learning_rate, int n) {
    float *d_params, *d_grads;

    // Allocate device memory
    cudaMalloc(&d_params, n * sizeof(float));
    cudaMalloc(&d_grads, n * sizeof(float));

    // Copy data to device
    cudaMemcpy(d_params, h_params, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_grads, h_grads, n * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    int threads_per_block = 256;
    int num_blocks = (n + threads_per_block - 1) / threads_per_block;
    sgd_update_kernel<<<num_blocks, threads_per_block>>>(d_params, d_grads,
                                                         learning_rate, n);

    cudaDeviceSynchronize();

    // Copy updated parameters back
    cudaMemcpy(h_params, d_params, n * sizeof(float), cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_params);
    cudaFree(d_grads);
}

// ================================
// Adam Update
// ================================
__global__ void adam_update_kernel(float* params, const float* grads,
                                  float* m, float* v,
                                  float learning_rate, float beta1, float beta2,
                                  float eps, int t, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float grad = grads[idx];

        // Update biased first moment estimate
        // m = beta1 * m + (1 - beta1) * grad
        m[idx] = beta1 * m[idx] + (1.0f - beta1) * grad;

        // Update biased second raw moment estimate
        // v = beta2 * v + (1 - beta2) * grad^2
        v[idx] = beta2 * v[idx] + (1.0f - beta2) * grad * grad;

        // Compute bias-corrected first moment estimate
        // m_hat = m / (1 - beta1^t)
        float m_hat = m[idx] / (1.0f - powf(beta1, t));

        // Compute bias-corrected second raw moment estimate
        // v_hat = v / (1 - beta2^t)
        float v_hat = v[idx] / (1.0f - powf(beta2, t));

        // Update parameters
        // param -= lr * m_hat / (sqrt(v_hat) + eps)
        params[idx] -= learning_rate * m_hat / (sqrtf(v_hat) + eps);
    }
}

void adam_update_cuda(float* h_params, const float* h_grads,
                     float* h_m, float* h_v,
                     float learning_rate, float beta1, float beta2,
                     float eps, int t, int n) {
    float *d_params, *d_grads, *d_m, *d_v;

    // Allocate device memory
    cudaMalloc(&d_params, n * sizeof(float));
    cudaMalloc(&d_grads, n * sizeof(float));
    cudaMalloc(&d_m, n * sizeof(float));
    cudaMalloc(&d_v, n * sizeof(float));

    // Copy data to device
    cudaMemcpy(d_params, h_params, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_grads, h_grads, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_m, h_m, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, h_v, n * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    int threads_per_block = 256;
    int num_blocks = (n + threads_per_block - 1) / threads_per_block;
    adam_update_kernel<<<num_blocks, threads_per_block>>>(
        d_params, d_grads, d_m, d_v,
        learning_rate, beta1, beta2, eps, t, n);

    cudaDeviceSynchronize();

    // Copy updated data back
    cudaMemcpy(h_params, d_params, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_m, d_m, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_v, d_v, n * sizeof(float), cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_params);
    cudaFree(d_grads);
    cudaFree(d_m);
    cudaFree(d_v);
}
