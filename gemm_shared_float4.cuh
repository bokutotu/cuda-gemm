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
__global__ void gemm_shared_float4(const float* __restrict__ A,
                            const float* __restrict__ B,
                            float* __restrict__ C,
                            int n, int m, int k) {
    __shared__ float As[TILE_SIZE][TILE_SIZE + 8];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE + 8];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int n_addr = blockIdx.x * TILE_SIZE + ty;
    int k_addr = blockIdx.y * TILE_SIZE + tx;

    float acc = 0.f;

    for (int i = 0; i < m; i += TILE_SIZE) {
        if (n_addr < n) {
            if ((tx & 3) == 0) {
                int a_col = i + tx;
                if (((m & 3) == 0) && (a_col + 3 < m)) {
                    const float4 v = *reinterpret_cast<const float4*>(
                        &A[n_addr * m + a_col]);
                    *reinterpret_cast<float4*>(&As[ty][tx]) = v;
                } else {
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        int col = a_col + j;
                        As[ty][tx + j] = (col < m) ? A[n_addr * m + col] : 0.f;
                    }
                }
            }
        } else if ((tx & 3) == 0) {
            #pragma unroll
            for (int j = 0; j < 4; ++j) As[ty][tx + j] = 0.f;
        }

        if ((i + ty) < m) {
            if ((tx & 3) == 0) {
                int b_col = k_addr;
                if (((k & 3) == 0) && (b_col + 3 < k)) {
                    const float4 v = *reinterpret_cast<const float4*>(
                        &B[(i + ty) * k + b_col]);
                    *reinterpret_cast<float4*>(&Bs[ty][tx]) = v;
                } else {
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        int col = b_col + j;
                        Bs[ty][tx + j] = (col < k) ? B[(i + ty) * k + col] : 0.f;
                    }
                }
            }
        } else if ((tx & 3) == 0) {
            #pragma unroll
            for (int j = 0; j < 4; ++j) Bs[ty][tx + j] = 0.f;
        }

        __syncthreads();

        #pragma unroll
        for (int j = 0; j < TILE_SIZE; ++j)
            acc += As[ty][j] * Bs[j][tx];

        __syncthreads();
    }

    if (n_addr < n && k_addr < k)
        C[n_addr * k + k_addr] = acc;
}
