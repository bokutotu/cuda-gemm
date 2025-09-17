#pragma once

#ifndef TILE_SIZE
#define TILE_SIZE 32
#endif

// Tiled (shared-memory) GEMM (row-major, BLAS-style)
// Computes: C[M,N] = A[M,K] x B[K,N]
// Shapes (row-major):
//  - A: M x K
//  - B: K x N
//  - C: M x N
__global__ void gemm_shared(const float* __restrict__ A,
                            const float* __restrict__ B,
                            float* __restrict__ C,
                            int m, int n, int k) {
    __shared__ float As[TILE_SIZE][TILE_SIZE + 1];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE + 1];

    // Block tiles cover M (rows) and N (cols)
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int m_row = blockIdx.x * TILE_SIZE + ty; // row in C/A (M)
    int n_col = blockIdx.y * TILE_SIZE + tx; // col in C/B (N)

    float acc = 0.f;
    // Iterate over K dimension in tiles
    for (int k0 = 0; k0 < k; k0 += TILE_SIZE) {
        if (m_row < m && (k0 + tx) < k)
            As[ty][tx] = A[m_row * k + (k0 + tx)];
        else
            As[ty][tx] = 0.f;
        if (n_col < n && (k0 + ty) < k)
            Bs[ty][tx] = B[(k0 + ty) * n + n_col];
        else
            Bs[ty][tx] = 0.f;

        __syncthreads();

        #pragma unroll
        for (int p = 0; p < TILE_SIZE; p++) acc += As[ty][p] * Bs[p][tx];
        __syncthreads();
    }
    if (m_row < m && n_col < n)
        C[m_row * n + n_col] = acc;
}
