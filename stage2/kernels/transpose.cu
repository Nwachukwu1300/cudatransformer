#include <cuda_runtime.h>
#include <stdio.h>

/**
 * Matrix Transpose
 *
 * Transposes a 2D matrix: output[j,i] = input[i,j]
 * Used in matmul backward pass (need A.T @ grad_output)
 *
 * Thread strategy: 16x16 blocks (TILE_SIZE x TILE_SIZE)
 * Uses shared memory tiling to improve performance and avoid bank conflicts
 */

#define TILE_SIZE 16

// ================================
// Transpose kernel with shared memory
// ================================
__global__ void transpose_kernel(const float* input, float* output,
                                int rows, int cols) {
    // Shared memory for tile (with padding to avoid bank conflicts)
    __shared__ float tile[TILE_SIZE][TILE_SIZE + 1];  // +1 for padding

    // Calculate global position
    int x = blockIdx.x * TILE_SIZE + threadIdx.x;
    int y = blockIdx.y * TILE_SIZE + threadIdx.y;

    // Load tile from input into shared memory
    if (y < rows && x < cols) {
        tile[threadIdx.y][threadIdx.x] = input[y * cols + x];
    }

    __syncthreads();

    // Calculate transposed position
    x = blockIdx.y * TILE_SIZE + threadIdx.x;
    y = blockIdx.x * TILE_SIZE + threadIdx.y;

    // Write transposed tile to output
    if (y < cols && x < rows) {
        output[y * rows + x] = tile[threadIdx.x][threadIdx.y];
    }
}

// Host wrapper for transpose
void transpose_cuda(const float* h_input, float* h_output,
                   int rows, int cols) {
    float *d_input, *d_output;

    // Allocate device memory
    cudaMalloc(&d_input, rows * cols * sizeof(float));
    cudaMalloc(&d_output, rows * cols * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, h_input, rows * cols * sizeof(float),
               cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threads_per_block(TILE_SIZE, TILE_SIZE);
    dim3 num_blocks((cols + TILE_SIZE - 1) / TILE_SIZE,
                   (rows + TILE_SIZE - 1) / TILE_SIZE);
    transpose_kernel<<<num_blocks, threads_per_block>>>(d_input, d_output,
                                                        rows, cols);

    // Synchronize
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(h_output, d_output, rows * cols * sizeof(float),
               cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
}
