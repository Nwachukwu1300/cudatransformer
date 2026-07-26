/*
 * Matrix Multiplication CUDA Kernel
 *
 * Computes C = A @ B where:
 *   A is (M x K)
 *   B is (K x N)
 *   C is (M x N)
 *
 * Parallelization strategy:
 * - 2D grid of 2D blocks
 * - Each thread computes one element of C
 * - C[row, col] = sum over k of A[row, k] * B[k, col]
 *
 * This is a naive implementation. For production, you'd use:
 * - Shared memory tiling to reduce global memory accesses
 * - Loop unrolling
 * - Register blocking
 *
 * But this version clearly shows the parallelization pattern.
 */

#include <cuda_runtime.h>

// Block size for 2D tiling
#define BLOCK_SIZE 16

// Naive kernel: one thread per output element
__global__ void matmul_kernel(const float* A, const float* B, float* C,
                              int M, int K, int N) {
    // Row and column of C element this thread computes
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;

        // Dot product of row of A with column of B
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }

        C[row * N + col] = sum;
    }
}

// Tiled kernel using shared memory (more efficient for large matrices)
__global__ void matmul_tiled_kernel(const float* A, const float* B, float* C,
                                    int M, int K, int N) {
    // Shared memory for tiles of A and B
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    float sum = 0.0f;

    // Loop over tiles
    for (int t = 0; t < (K + BLOCK_SIZE - 1) / BLOCK_SIZE; t++) {
        // Load tile of A into shared memory
        int a_col = t * BLOCK_SIZE + threadIdx.x;
        if (row < M && a_col < K) {
            As[threadIdx.y][threadIdx.x] = A[row * K + a_col];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Load tile of B into shared memory
        int b_row = t * BLOCK_SIZE + threadIdx.y;
        if (b_row < K && col < N) {
            Bs[threadIdx.y][threadIdx.x] = B[b_row * N + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Synchronize to ensure tile is loaded
        __syncthreads();

        // Compute partial dot product for this tile
        for (int k = 0; k < BLOCK_SIZE; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        // Synchronize before loading next tile
        __syncthreads();
    }

    // Write result
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// Host function to launch the kernel
void matmul_cuda(const float* h_A, const float* h_B, float* h_C,
                 int M, int K, int N, bool use_tiled) {
    float *d_A, *d_B, *d_C;

    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    // Allocate device memory
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);

    // Copy inputs to device
    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

    // Configure grid and block dimensions
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
              (M + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Launch kernel
    if (use_tiled) {
        matmul_tiled_kernel<<<grid, block>>>(d_A, d_B, d_C, M, K, N);
    } else {
        matmul_kernel<<<grid, block>>>(d_A, d_B, d_C, M, K, N);
    }

    cudaDeviceSynchronize();

    // Copy result back
    cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}
