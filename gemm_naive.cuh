// Naive matrix multiplication (row-major, BLAS-style)
// Computes: C[M,N] = A[M,K] x B[K,N]
// Shapes (row-major):
//  - A: M x K
//  - B: K x N
//  - C: M x N
#pragma once

__global__ void gemm_naive(const float* __restrict__ A,
                           const float* __restrict__ B,
                           float* __restrict__ C,
                           int m, int n, int k) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x; // N-dimension (columns)
    const int row = blockIdx.y * blockDim.y + threadIdx.y; // M-dimension (rows)

    if (row >= m || col >= n) return;

    float acc = 0.0f;
    for (int p = 0; p < k; ++p) {
        const float a = A[row * k + p];   // A[row, p]
        const float b = B[p * n + col];   // B[p, col]
        acc += a * b;
    }

    C[row * n + col] = acc;
}
