// Naive matrix multiplication (row-major)
// Computes: C[n,k] = A[n,m] x B[m,k]
// Shapes (row-major):
//  - A: n x m
//  - B: m x k
//  - C: n x k
#pragma once

__global__ void gemm_naive(const float* __restrict__ A,
                           const float* __restrict__ B,
                           float* __restrict__ C,
                           int n, int m, int k) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x; // k-dimension (columns)
    const int row = blockIdx.y * blockDim.y + threadIdx.y; // n-dimension (rows)

    if (row >= n || col >= k) return;

    float acc = 0.0f;
    for (int p = 0; p < m; ++p) {
        const float a = A[row * m + p];   // A[row, p]
        const float b = B[p * k + col];   // B[p, col]
        acc += a * b;
    }

    C[row * k + col] = acc;
}
