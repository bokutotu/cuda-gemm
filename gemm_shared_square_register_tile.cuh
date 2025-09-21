#pragma once

// Tiled (shared-memory) GEMM (row-major, BLAS-style)
// Computes: C[M,N] = A[M,K] x B[K,N]
// Shapes (row-major):
//  - A: M x K
//  - B: K x N
//  - C: M x N
// SHARED TILE SIZE (BMxBN)
// REGISTER TILE SIZE (RMxRN)
// blockDim.x -> BM / RM
// blockDIm.y -> BN / RN
// gridDim -> m / BM
// gridDim -> n / BN
template <int BM = 32, int BN = 32, int BK = 32, int RM = 4, int RN = 4>
__global__ void gemm_shared_square_register_tile(const float* __restrict__ A,
                            const float* __restrict__ B,
                            float* __restrict__ C,
                            int m, int n, int k) {
    __shared__ float As[BM][BK + 1];
    __shared__ float Bs[BK][BN + 1];

    // Block tiles cover M (rows) and N (cols)
    int tx = threadIdx.x; // x -> mに対応
    int ty = threadIdx.y; // y -> nに対応
    int m_row = blockIdx.x * BM + tx * RM; // row in C/A (M)
    int n_col = blockIdx.y * BN + ty * RN; // col in C/B (N)

    float acc[RM][RN];
    #pragma unroll
    for (int m = 0; m < RM; m++)
        #pragma unroll
        for (int n = 0; n < RN; n++)
            acc[m][n] = 0.;


    const int stride_k_A = BN; // blockDim.y * RN
    const int stride_k_B = BM; // blockDim.x * RM

    // Iterate over K dimension in tiles
    for (int k0 = 0; k0 < k; k0 += BK) {
        // Load A tile (BM x BK)
        for (int kk_base = ty * RN; kk_base < BK; kk_base += stride_k_A) {
            #pragma unroll
            for (int tm = 0; tm < RM; tm++) {
                int shared_row = tx * RM + tm;
                int global_row = m_row + tm;
                if (shared_row >= BM) continue;
                #pragma unroll
                for (int tn = 0; tn < RN; tn++) {
                    int kk = kk_base + tn;
                    if (kk >= BK) break;
                    float val = 0.f;
                    if (global_row < m && (k0 + kk) < k)
                        val = A[static_cast<size_t>(global_row) * k + (k0 + kk)];
                    As[shared_row][kk] = val;
                }
            }
        }

        // Load B tile (BK x BN)
        for (int kk_base = tx * RM; kk_base < BK; kk_base += stride_k_B) {
            #pragma unroll
            for (int tm = 0; tm < RM; tm++) {
                int kk = kk_base + tm;
                if (kk >= BK) break;
                int global_k = k0 + kk;
                #pragma unroll
                for (int tn = 0; tn < RN; tn++) {
                    int shared_col = ty * RN + tn;
                    if (shared_col >= BN) continue;
                    float val = 0.f;
                    int global_col = n_col + tn;
                    if (global_k < k && global_col < n)
                        val = B[static_cast<size_t>(global_k) * n + global_col];
                    Bs[kk][shared_col] = val;
                }
            }
        }


        __syncthreads();

        #pragma unroll
        for (int p = 0; p < BK; p++)
            #pragma unroll
            for (int pn = 0; pn < RN; pn++)
                #pragma unroll
                for (int pm = 0; pm < RM; pm++)
                    acc[pm][pn] += As[tx * RM + pm][p] * Bs[p][ty * RN + pn];
        __syncthreads();
    }
    if (m_row < m && n_col < n) {
        #pragma unroll
        for (int rm_idx = 0; rm_idx < RM; rm_idx++) {
            int row = m_row + rm_idx;
            if (row >= m) break;
            #pragma unroll
            for (int rn_idx = 0; rn_idx < RN; rn_idx++) {
                int col = n_col + rn_idx;
                if (col >= n) break;
                C[row * n + col] = acc[rm_idx][rn_idx];
            }
        }
    }
}
