#pragma once

#ifndef TILE_SIZE
#define TILE_SIZE 32
#endif

// Tiled (shared-memory) matrix multiplication (row-major)
// Computes: C[n,k] = A[n,m] x B[m,k]
// Shapes (row-major):
//  - A: n x m
//  - B: m x k
//  - C: n x k
__global__ void gemm_shared(const float* __restrict__ A,
                            const float* __restrict__ B,
                            float* __restrict__ C,
                            int n, int m, int k) {
    __shared__ float As[TILE_SIZE][TILE_SIZE + 1];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE + 1];

    // Block tiles cover n (rows) and k (cols)
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int n_addr = blockIdx.x * TILE_SIZE + ty; // row in C/A
    int k_addr = blockIdx.y * TILE_SIZE + tx; // col in C/B

    float acc = 0.f;
    // Iterate over m dimension in tiles
    for (int i = 0; i < m; i += TILE_SIZE) {
        if (n_addr < n && (i + tx) < m)
            As[ty][tx] = A[n_addr * m + (i + tx)];
        else
            As[ty][tx] = 0.f;
        if (k_addr < k && (i + ty) < m)
            Bs[ty][tx] = B[(i + ty) * k + k_addr];
        else
            Bs[ty][tx] = 0.f;

        __syncthreads();

        #pragma unroll
        for (int j = 0; j < TILE_SIZE; j++) acc += As[ty][j] * Bs[j][tx];
        __syncthreads();
    }
    if (n_addr < n && k_addr < k)
        C[n_addr * k + k_addr] = acc;
}
