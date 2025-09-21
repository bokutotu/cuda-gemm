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

    const int threads_per_block = blockDim.x * blockDim.y;
    const int lane_id = ty * blockDim.x + tx;

    float acc[RM][RN];
    #pragma unroll
    for (int m = 0; m < RM; m++)
        #pragma unroll
        for (int n = 0; n < RN; n++)
            acc[m][n] = 0.;


    // Iterate over K dimension in tiles
    for (int k0 = 0; k0 < k; k0 += BK) {
        // Load A tile (BM x BK)
        for (int idx = lane_id; idx < BM * BK; idx += threads_per_block) {
            int local_row = idx / BK;
            int local_col = idx % BK;
            int global_row = blockIdx.x * BM + local_row;
            int global_col = k0 + local_col;
            float val = 0.f;
            if (global_row < m && global_col < k)
                val = A[static_cast<size_t>(global_row) * k + global_col];
            As[local_row][local_col] = val;
        }

        // Load B tile (BK x BN)
        for (int idx = lane_id; idx < BK * BN; idx += threads_per_block) {
            int local_row = idx / BN;
            int local_col = idx % BN;
            int global_row = k0 + local_row;
            int global_col = blockIdx.y * BN + local_col;
            float val = 0.f;
            if (global_row < k && global_col < n)
                val = B[static_cast<size_t>(global_row) * n + global_col];
            Bs[local_row][local_col] = val;
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
