#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

/**
 * Reduction Operations
 *
 * Implements reduction operations along specified axes:
 * - reduce_mean: Mean along last axis (for LayerNorm)
 * - reduce_variance: Variance along last axis (for LayerNorm)
 * - bias_add: Add 1D bias to 2D matrix (broadcast across rows)
 *
 * Thread strategy: One block per row for reductions, 16x16 blocks for bias_add
 */

// ================================
// Reduce Mean (along last axis)
// ================================
// Each block handles one row
__global__ void reduce_mean_kernel(const float* input, float* output,
                                  int num_rows, int num_cols) {
    int row = blockIdx.x;
    if (row >= num_rows) return;

    // Shared memory for partial sums
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int idx = row * num_cols + tid;

    // Load data and compute partial sum
    float sum = 0.0f;
    for (int i = tid; i < num_cols; i += blockDim.x) {
        sum += input[row * num_cols + i];
    }
    sdata[tid] = sum;
    __syncthreads();

    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write result
    if (tid == 0) {
        output[row] = sdata[0] / num_cols;
    }
}

void reduce_mean_cuda(const float* h_input, float* h_output,
                     int num_rows, int num_cols) {
    float *d_input, *d_output;

    cudaMalloc(&d_input, num_rows * num_cols * sizeof(float));
    cudaMalloc(&d_output, num_rows * sizeof(float));

    cudaMemcpy(d_input, h_input, num_rows * num_cols * sizeof(float),
               cudaMemcpyHostToDevice);

    // One block per row, threads per block depends on num_cols
    int threads = min(256, (int)pow(2, ceil(log2(num_cols))));
    reduce_mean_kernel<<<num_rows, threads, threads * sizeof(float)>>>(
        d_input, d_output, num_rows, num_cols);

    cudaDeviceSynchronize();

    cudaMemcpy(h_output, d_output, num_rows * sizeof(float),
               cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
}

// ================================
// Reduce Variance (along last axis)
// ================================
// Computes variance given input and pre-computed mean
__global__ void reduce_variance_kernel(const float* input, const float* mean,
                                      float* output, int num_rows, int num_cols) {
    int row = blockIdx.x;
    if (row >= num_rows) return;

    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    float row_mean = mean[row];

    // Compute partial sum of squared differences
    float sum = 0.0f;
    for (int i = tid; i < num_cols; i += blockDim.x) {
        float diff = input[row * num_cols + i] - row_mean;
        sum += diff * diff;
    }
    sdata[tid] = sum;
    __syncthreads();

    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write result (variance)
    if (tid == 0) {
        output[row] = sdata[0] / num_cols;
    }
}

void reduce_variance_cuda(const float* h_input, const float* h_mean,
                         float* h_output, int num_rows, int num_cols) {
    float *d_input, *d_mean, *d_output;

    cudaMalloc(&d_input, num_rows * num_cols * sizeof(float));
    cudaMalloc(&d_mean, num_rows * sizeof(float));
    cudaMalloc(&d_output, num_rows * sizeof(float));

    cudaMemcpy(d_input, h_input, num_rows * num_cols * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_mean, h_mean, num_rows * sizeof(float),
               cudaMemcpyHostToDevice);

    int threads = min(256, (int)pow(2, ceil(log2(num_cols))));
    reduce_variance_kernel<<<num_rows, threads, threads * sizeof(float)>>>(
        d_input, d_mean, d_output, num_rows, num_cols);

    cudaDeviceSynchronize();

    cudaMemcpy(h_output, d_output, num_rows * sizeof(float),
               cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_mean);
    cudaFree(d_output);
}

// ================================
// Bias Add (broadcast 1D bias to 2D matrix)
// ================================
// output[i,j] = input[i,j] + bias[j]
__global__ void bias_add_kernel(const float* input, const float* bias,
                               float* output, int num_rows, int num_cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < num_rows && col < num_cols) {
        output[row * num_cols + col] = input[row * num_cols + col] + bias[col];
    }
}

void bias_add_cuda(const float* h_input, const float* h_bias,
                  float* h_output, int num_rows, int num_cols) {
    float *d_input, *d_bias, *d_output;

    cudaMalloc(&d_input, num_rows * num_cols * sizeof(float));
    cudaMalloc(&d_bias, num_cols * sizeof(float));
    cudaMalloc(&d_output, num_rows * num_cols * sizeof(float));

    cudaMemcpy(d_input, h_input, num_rows * num_cols * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, h_bias, num_cols * sizeof(float),
               cudaMemcpyHostToDevice);

    // Use 16x16 blocks
    dim3 threads_per_block(16, 16);
    dim3 num_blocks((num_cols + 15) / 16, (num_rows + 15) / 16);
    bias_add_kernel<<<num_blocks, threads_per_block>>>(d_input, d_bias,
                                                       d_output, num_rows, num_cols);

    cudaDeviceSynchronize();

    cudaMemcpy(h_output, d_output, num_rows * num_cols * sizeof(float),
               cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_bias);
    cudaFree(d_output);
}
