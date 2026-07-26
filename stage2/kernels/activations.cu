#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

/**
 * Activation Functions
 *
 * Implements ReLU and GELU activations with their backward passes
 *
 * ReLU: f(x) = max(0, x)
 * ReLU': f'(x) = 1 if x > 0 else 0
 *
 * GELU: f(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
 * GELU': Complex derivative involving tanh and sech²
 *
 * Thread strategy: 256 threads/block, 1D grid
 */

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define GELU_COEFF_A 0.044715f
#define SQRT_2_OVER_PI 0.79788456080286535587989f  // sqrt(2/π)

// ================================
// ReLU Forward
// ================================
__global__ void relu_forward_kernel(const float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = fmaxf(0.0f, input[idx]);
    }
}

void relu_forward_cuda(const float* h_input, float* h_output, int n) {
    float *d_input, *d_output;

    cudaMalloc(&d_input, n * sizeof(float));
    cudaMalloc(&d_output, n * sizeof(float));

    cudaMemcpy(d_input, h_input, n * sizeof(float), cudaMemcpyHostToDevice);

    int threads_per_block = 256;
    int num_blocks = (n + threads_per_block - 1) / threads_per_block;
    relu_forward_kernel<<<num_blocks, threads_per_block>>>(d_input, d_output, n);

    cudaDeviceSynchronize();

    cudaMemcpy(h_output, d_output, n * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
}

// ================================
// ReLU Backward
// ================================
__global__ void relu_backward_kernel(const float* grad_output, const float* input,
                                     float* grad_input, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Gradient passes through if input > 0, otherwise it's 0
        grad_input[idx] = (input[idx] > 0.0f) ? grad_output[idx] : 0.0f;
    }
}

void relu_backward_cuda(const float* h_grad_output, const float* h_input,
                        float* h_grad_input, int n) {
    float *d_grad_output, *d_input, *d_grad_input;

    cudaMalloc(&d_grad_output, n * sizeof(float));
    cudaMalloc(&d_input, n * sizeof(float));
    cudaMalloc(&d_grad_input, n * sizeof(float));

    cudaMemcpy(d_grad_output, h_grad_output, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_input, h_input, n * sizeof(float), cudaMemcpyHostToDevice);

    int threads_per_block = 256;
    int num_blocks = (n + threads_per_block - 1) / threads_per_block;
    relu_backward_kernel<<<num_blocks, threads_per_block>>>(d_grad_output, d_input,
                                                            d_grad_input, n);

    cudaDeviceSynchronize();

    cudaMemcpy(h_grad_input, d_grad_input, n * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_grad_output);
    cudaFree(d_input);
    cudaFree(d_grad_input);
}

// ================================
// GELU Forward
// ================================
__global__ void gelu_forward_kernel(const float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = input[idx];
        // φ(x) = sqrt(2/π) * (x + 0.044715 * x³)
        float phi = SQRT_2_OVER_PI * (x + GELU_COEFF_A * x * x * x);
        // GELU(x) = 0.5 * x * (1 + tanh(φ(x)))
        output[idx] = 0.5f * x * (1.0f + tanhf(phi));
    }
}

void gelu_forward_cuda(const float* h_input, float* h_output, int n) {
    float *d_input, *d_output;

    cudaMalloc(&d_input, n * sizeof(float));
    cudaMalloc(&d_output, n * sizeof(float));

    cudaMemcpy(d_input, h_input, n * sizeof(float), cudaMemcpyHostToDevice);

    int threads_per_block = 256;
    int num_blocks = (n + threads_per_block - 1) / threads_per_block;
    gelu_forward_kernel<<<num_blocks, threads_per_block>>>(d_input, d_output, n);

    cudaDeviceSynchronize();

    cudaMemcpy(h_output, d_output, n * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
}

// ================================
// GELU Backward
// ================================
__global__ void gelu_backward_kernel(const float* grad_output, const float* input,
                                     float* grad_input, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = input[idx];

        // φ(x) = sqrt(2/π) * (x + 0.044715 * x³)
        float phi = SQRT_2_OVER_PI * (x + GELU_COEFF_A * x * x * x);

        // φ'(x) = sqrt(2/π) * (1 + 3 * 0.044715 * x²)
        float phi_prime = SQRT_2_OVER_PI * (1.0f + 3.0f * GELU_COEFF_A * x * x);

        // tanh(φ)
        float tanh_phi = tanhf(phi);

        // sech²(φ) = 1 - tanh²(φ)
        float sech2_phi = 1.0f - tanh_phi * tanh_phi;

        // GELU'(x) = 0.5 * (1 + tanh(φ)) + 0.5 * x * sech²(φ) * φ'(x)
        float gelu_derivative = 0.5f * (1.0f + tanh_phi) +
                                0.5f * x * sech2_phi * phi_prime;

        grad_input[idx] = grad_output[idx] * gelu_derivative;
    }
}

void gelu_backward_cuda(const float* h_grad_output, const float* h_input,
                        float* h_grad_input, int n) {
    float *d_grad_output, *d_input, *d_grad_input;

    cudaMalloc(&d_grad_output, n * sizeof(float));
    cudaMalloc(&d_input, n * sizeof(float));
    cudaMalloc(&d_grad_input, n * sizeof(float));

    cudaMemcpy(d_grad_output, h_grad_output, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_input, h_input, n * sizeof(float), cudaMemcpyHostToDevice);

    int threads_per_block = 256;
    int num_blocks = (n + threads_per_block - 1) / threads_per_block;
    gelu_backward_kernel<<<num_blocks, threads_per_block>>>(d_grad_output, d_input,
                                                            d_grad_input, n);

    cudaDeviceSynchronize();

    cudaMemcpy(h_grad_input, d_grad_input, n * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_grad_output);
    cudaFree(d_input);
    cudaFree(d_grad_input);
}
