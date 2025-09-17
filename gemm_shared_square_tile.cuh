#pragma once

// Tiled (shared-memory) GEMM (row-major, BLAS-style)
// Computes: C[M,N] = A[M,K] x B[K,N]
// Shapes (row-major):
//  - A: M x K
//  - B: K x N
//  - C: M x N
template <int BM = 32, int BN = 32, int BK = 32>
__global__ void gemm_shared_square(const float* __restrict__ A,
                            const float* __restrict__ B,
                            float* __restrict__ C,
                            int m, int n, int k) {
    __shared__ float As[BM][BK + 1];
    __shared__ float Bs[BK][BN + 1];

    // Block tiles cover M (rows) and N (cols)
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int m_row = blockIdx.x * BM + ty; // row in C/A (M)
    int n_col = blockIdx.y * BN + tx; // col in C/B (N)

    float acc = 0.f;
    constexpr int A_TILE = (BK + BN - 1) / BN;
    constexpr int B_TILE = (BK + BM - 1) / BM;
    // Iterate over K dimension in tiles
    for (int k0 = 0; k0 < k; k0 += BK) {
        #pragma unroll
        for (int t = 0; t < A_TILE; t++) {
            int kk = t * BN + tx;
            if (kk >= BK) break;
            if (m_row < m && (kk + k0)< k)
                As[ty][kk] = A[m_row * k + (k0 + kk)];
            else
                As[ty][kk] = 0.f;
        }

        #pragma unroll
        for (int t = 0; t < B_TILE; t++) {
            int kk = ty + t * BM;
            if (kk >= BK) break;
            if (n_col < n && (kk + k0) < k)
                Bs[kk][tx] = B[(kk + k0) * n + n_col];
            else
                Bs[kk][tx] = 0.f;
        }

        __syncthreads();

        #pragma unroll
        for (int p = 0; p < BK; p++) acc += As[ty][p] * Bs[p][tx];
        __syncthreads();
    }
    if (m_row < m && n_col < n)
        C[m_row * n + n_col] = acc;
}
