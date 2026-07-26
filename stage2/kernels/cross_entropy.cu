#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

/**
 * Cross Entropy Loss
 *
 * Implements fused cross entropy loss computation:
 * - Forward: Computes -sum(target * log(softmax(logits))) / batch_size
 * - Backward: Returns (softmax(logits) - target) / batch_size
 *
 * Uses log-sum-exp trick for numerical stability
 *
 * Thread strategy: 16x16 blocks for forward, one block per sample for softmax
 */

// ================================
// Cross Entropy Forward
// ================================
// Helper: Compute max along rows (for numerical stability)
__global__ void row_max_kernel(const float* input, float* output,
                              int num_rows, int num_cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= num_rows) return;

    float max_val = -INFINITY;
    for (int col = 0; col < num_cols; col++) {
        max_val = fmaxf(max_val, input[row * num_cols + col]);
    }
    output[row] = max_val;
}

// Helper: Compute log-sum-exp along rows
__global__ void log_sum_exp_kernel(const float* input, const float* max_vals,
                                  float* output, int num_rows, int num_cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= num_rows) return;

    float max_val = max_vals[row];
    float sum = 0.0f;
    for (int col = 0; col < num_cols; col++) {
        sum += expf(input[row * num_cols + col] - max_val);
    }
    output[row] = max_val + logf(sum);
}

// Compute cross entropy loss
__global__ void cross_entropy_kernel(const float* logits, const float* targets,
                                    const float* log_sum_exp, float* losses,
                                    int num_rows, int num_cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < num_rows && col < num_cols) {
        int idx = row * num_cols + col;
        // loss = -target * (logit - log_sum_exp)
        losses[idx] = -targets[idx] * (logits[idx] - log_sum_exp[row]);
    }
}

void cross_entropy_forward_cuda(const float* h_logits, const float* h_targets,
                               float* h_loss, int num_samples, int num_classes) {
    float *d_logits, *d_targets, *d_max_vals, *d_log_sum_exp, *d_losses;

    // Allocate device memory
    cudaMalloc(&d_logits, num_samples * num_classes * sizeof(float));
    cudaMalloc(&d_targets, num_samples * num_classes * sizeof(float));
    cudaMalloc(&d_max_vals, num_samples * sizeof(float));
    cudaMalloc(&d_log_sum_exp, num_samples * sizeof(float));
    cudaMalloc(&d_losses, num_samples * num_classes * sizeof(float));

    // Copy inputs to device
    cudaMemcpy(d_logits, h_logits, num_samples * num_classes * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_targets, h_targets, num_samples * num_classes * sizeof(float),
               cudaMemcpyHostToDevice);

    // Step 1: Find max in each row
    int threads = min(256, num_samples);
    int blocks = (num_samples + threads - 1) / threads;
    row_max_kernel<<<blocks, threads>>>(d_logits, d_max_vals,
                                       num_samples, num_classes);

    // Step 2: Compute log-sum-exp
    log_sum_exp_kernel<<<blocks, threads>>>(d_logits, d_max_vals,
                                           d_log_sum_exp, num_samples, num_classes);

    // Step 3: Compute cross entropy
    dim3 threads_2d(16, 16);
    dim3 blocks_2d((num_classes + 15) / 16, (num_samples + 15) / 16);
    cross_entropy_kernel<<<blocks_2d, threads_2d>>>(d_logits, d_targets,
                                                    d_log_sum_exp, d_losses,
                                                    num_samples, num_classes);

    cudaDeviceSynchronize();

    // Copy result back and sum
    float* h_losses = new float[num_samples * num_classes];
    cudaMemcpy(h_losses, d_losses, num_samples * num_classes * sizeof(float),
               cudaMemcpyDeviceToHost);

    // Sum all losses and divide by batch size
    float total_loss = 0.0f;
    for (int i = 0; i < num_samples * num_classes; i++) {
        total_loss += h_losses[i];
    }
    *h_loss = total_loss / num_samples;

    // Cleanup
    delete[] h_losses;
    cudaFree(d_logits);
    cudaFree(d_targets);
    cudaFree(d_max_vals);
    cudaFree(d_log_sum_exp);
    cudaFree(d_losses);
}

// ================================
// Cross Entropy Backward
// ================================
// Gradient: (softmax(logits) - target) / batch_size
__global__ void cross_entropy_backward_kernel(const float* logits,
                                             const float* targets,
                                             const float* log_sum_exp,
                                             float* grad_logits,
                                             int num_rows, int num_cols,
                                             float batch_size) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < num_rows && col < num_cols) {
        int idx = row * num_cols + col;
        // softmax(logit) = exp(logit - log_sum_exp)
        float softmax_val = expf(logits[idx] - log_sum_exp[row]);
        // gradient = (softmax - target) / batch_size
        grad_logits[idx] = (softmax_val - targets[idx]) / batch_size;
    }
}

void cross_entropy_backward_cuda(const float* h_logits, const float* h_targets,
                                float* h_grad_logits,
                                int num_samples, int num_classes) {
    float *d_logits, *d_targets, *d_max_vals, *d_log_sum_exp, *d_grad_logits;

    // Allocate device memory
    cudaMalloc(&d_logits, num_samples * num_classes * sizeof(float));
    cudaMalloc(&d_targets, num_samples * num_classes * sizeof(float));
    cudaMalloc(&d_max_vals, num_samples * sizeof(float));
    cudaMalloc(&d_log_sum_exp, num_samples * sizeof(float));
    cudaMalloc(&d_grad_logits, num_samples * num_classes * sizeof(float));

    // Copy inputs to device
    cudaMemcpy(d_logits, h_logits, num_samples * num_classes * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_targets, h_targets, num_samples * num_classes * sizeof(float),
               cudaMemcpyHostToDevice);

    // Step 1: Find max in each row
    int threads = min(256, num_samples);
    int blocks = (num_samples + threads - 1) / threads;
    row_max_kernel<<<blocks, threads>>>(d_logits, d_max_vals,
                                       num_samples, num_classes);

    // Step 2: Compute log-sum-exp
    log_sum_exp_kernel<<<blocks, threads>>>(d_logits, d_max_vals,
                                           d_log_sum_exp, num_samples, num_classes);

    // Step 3: Compute gradient
    dim3 threads_2d(16, 16);
    dim3 blocks_2d((num_classes + 15) / 16, (num_samples + 15) / 16);
    cross_entropy_backward_kernel<<<blocks_2d, threads_2d>>>(
        d_logits, d_targets, d_log_sum_exp, d_grad_logits,
        num_samples, num_classes, (float)num_samples);

    cudaDeviceSynchronize();

    // Copy result back
    cudaMemcpy(h_grad_logits, d_grad_logits,
               num_samples * num_classes * sizeof(float),
               cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_logits);
    cudaFree(d_targets);
    cudaFree(d_max_vals);
    cudaFree(d_log_sum_exp);
    cudaFree(d_grad_logits);
}
