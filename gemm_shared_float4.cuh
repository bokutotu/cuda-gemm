#pragma once

#ifndef TILE_SIZE
#define TILE_SIZE 32
#endif

// Tiled (shared-memory) GEMM with float4 vectorized global loads (row-major, BLAS-style)
// Computes: C[M,N] = A[M,K] x B[K,N]
// Shapes (row-major):
//  - A: M x K
//  - B: K x N
//  - C: M x N
__global__ void gemm_shared_float4(const float* __restrict__ A,
                            const float* __restrict__ B,
                            float* __restrict__ C,
                            int m, int n, int k) {
    __shared__ float As[TILE_SIZE][TILE_SIZE + 8];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE + 8];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int m_row = blockIdx.x * TILE_SIZE + ty; // row in C/A (M)
    int n_col = blockIdx.y * TILE_SIZE + tx; // col in C/B (N)

    float acc = 0.f;

    for (int k0 = 0; k0 < k; k0 += TILE_SIZE) {
        // Load A tile: vectorize along K (contiguous)
        if (m_row < m) {
            if ((tx & 3) == 0) {
                int a_col = k0 + tx;
                if (((k & 3) == 0) && (a_col + 3 < k)) {
                    const float4 v = *reinterpret_cast<const float4*>(
                        &A[m_row * k + a_col]);
                    *reinterpret_cast<float4*>(&As[ty][tx]) = v;
                } else {
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        int col = a_col + j;
                        As[ty][tx + j] = (col < k) ? A[m_row * k + col] : 0.f;
                    }
                }
            }
        } else if ((tx & 3) == 0) {
            #pragma unroll
            for (int j = 0; j < 4; ++j) As[ty][tx + j] = 0.f;
        }

        // Load B tile: vectorize along N (contiguous)
        if ((k0 + ty) < k) {
            if ((tx & 3) == 0) {
                int b_col = n_col;
                if (((n & 3) == 0) && (b_col + 3 < n)) {
                    const float4 v = *reinterpret_cast<const float4*>(
                        &B[(k0 + ty) * n + b_col]);
                    *reinterpret_cast<float4*>(&Bs[ty][tx]) = v;
                } else {
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        int col = b_col + j;
                        Bs[ty][tx + j] = (col < n) ? B[(k0 + ty) * n + col] : 0.f;
                    }
                }
            }
        } else if ((tx & 3) == 0) {
            #pragma unroll
            for (int j = 0; j < 4; ++j) Bs[ty][tx + j] = 0.f;
        }

        __syncthreads();

        #pragma unroll
        for (int p = 0; p < TILE_SIZE; ++p)
            acc += As[ty][p] * Bs[p][tx];

        __syncthreads();
    }

    if (m_row < m && n_col < n)
        C[m_row * n + n_col] = acc;
}
