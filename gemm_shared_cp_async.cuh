#pragma once

#ifndef TILE_SIZE
#define TILE_SIZE 32
#endif

// cp.async-based ping-pong GEMM for SM80+
// Row-major, BLAS-style: C[M,N] = A[M,K] x B[K,N]
// Each thread computes one C element; tiles are TILE_SIZE x TILE_SIZE.
// Uses cp.async to overlap global->shared copies for tile t+1 while computing tile t.
__global__ void gemm_shared_cp_async(const float* __restrict__ A,
                                     const float* __restrict__ B,
                                     float* __restrict__ C,
                                     int m, int n, int k) {
#if __CUDA_ARCH__ >= 800
    // Use +8 padding so each row stride is 160B for TILE_SIZE=32,
    // which preserves 16B alignment for cp.async. Also mitigates bank conflicts.
    __shared__ float As[2][TILE_SIZE][TILE_SIZE + 8];
    __shared__ float Bs[2][TILE_SIZE][TILE_SIZE + 8];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int m_row = blockIdx.x * TILE_SIZE + ty; // row in C/A (M)
    const int n_col = blockIdx.y * TILE_SIZE + tx; // col in C/B (N)

    float acc = 0.f;

    int buf = 0;

    // Helper lambdas for cp.async copies with edge handling
    auto cp_async_f32 = [](uint32_t dst_smem, const float* src_gmem, bool pred) {
        if (pred) {
            asm volatile(
                "cp.async.ca.shared.global [%0], [%1], %2;\n" ::
                "r"(dst_smem), "l"(src_gmem), "n"(4));
        } else {
            // Fallback: write zero to smem for out-of-bounds
            asm volatile("st.shared.f32 [%0], %1;\n" :: "r"(dst_smem), "f"(0.0f));
        }
    };
    auto cp_async_f16 = [](uint32_t dst_smem, const float* src_gmem, bool pred) {
        if (pred) {
            asm volatile(
                "cp.async.ca.shared.global [%0], [%1], %2;\n" ::
                "r"(dst_smem), "l"(src_gmem), "n"(16));
        } else {
            // Zero-fill 16B (4 floats)
            asm volatile("st.shared.f32 [%0], %1;\n" :: "r"(dst_smem + 0),  "f"(0.0f));
            asm volatile("st.shared.f32 [%0], %1;\n" :: "r"(dst_smem + 4),  "f"(0.0f));
            asm volatile("st.shared.f32 [%0], %1;\n" :: "r"(dst_smem + 8),  "f"(0.0f));
            asm volatile("st.shared.f32 [%0], %1;\n" :: "r"(dst_smem + 12), "f"(0.0f));
        }
    };

    // Prime tile 0 into buffer 0
    {
        const int a_col0 = 0 + tx;  // along K
        const int b_row0 = 0 + ty;  // along K
        uint32_t sA = __cvta_generic_to_shared(&As[buf][ty][tx]);
        uint32_t sB = __cvta_generic_to_shared(&Bs[buf][ty][tx]);
        if ((tx & 3) == 0) {
            // Try 16B vectorized copy when aligned and fully in-bounds
            const bool canA16 = ((k & 3) == 0) && (m_row < m) && (a_col0 + 3 < k);
            const bool canB16 = ((n & 3) == 0) && (b_row0 < k) && (n_col + 3 < n);
            const float* gA = A + static_cast<size_t>(m_row) * k + a_col0;
            const float* gB = B + static_cast<size_t>(b_row0) * n + n_col;
            cp_async_f16(sA, gA, canA16);
            cp_async_f16(sB, gB, canB16);
            if (!canA16) {
                // Fallback element-wise (covers edges)
                #pragma unroll
                for (int jj = 0; jj < 4; ++jj) {
                    const int acol = a_col0 + jj;
                    cp_async_f32(sA + jj * 4, A + static_cast<size_t>(m_row) * k + acol,
                                 (m_row < m) && (acol < k));
                }
            }
            if (!canB16) {
                #pragma unroll
                for (int jj = 0; jj < 4; ++jj) {
                    const int ncol = n_col + jj;
                    cp_async_f32(sB + jj * 4, B + static_cast<size_t>(b_row0) * n + ncol,
                                 (b_row0 < k) && (ncol < n));
                }
            }
        }
        asm volatile("cp.async.commit_group;\n" ::);
        asm volatile("cp.async.wait_group 0;\n" ::);
        __syncthreads();
    }

    // Main loop over tiles with pipelined copies and compute
    for (int k0 = 0; k0 < k; k0 += TILE_SIZE) {
        const int next = buf ^ 1;
        const int inext = k0 + TILE_SIZE;

        // Issue async copies for next tile (if any), overlapping with compute below
        if (inext < k) {
            const int a_col = inext + tx; // along K
            const int b_row = inext + ty; // along K

            uint32_t sA2 = __cvta_generic_to_shared(&As[next][ty][tx]);
            uint32_t sB2 = __cvta_generic_to_shared(&Bs[next][ty][tx]);
            if ((tx & 3) == 0) {
                const bool canA16 = ((k & 3) == 0) && (m_row < m) && (a_col + 3 < k);
                const bool canB16 = ((n & 3) == 0) && (b_row < k) && (n_col + 3 < n);
                const float* gA = A + static_cast<size_t>(m_row) * k + a_col;
                const float* gB = B + static_cast<size_t>(b_row) * n + n_col;
                cp_async_f16(sA2, gA, canA16);
                cp_async_f16(sB2, gB, canB16);
                if (!canA16) {
                    #pragma unroll
                    for (int jj = 0; jj < 4; ++jj) {
                        const int acol = a_col + jj;
                        cp_async_f32(sA2 + jj * 4, A + static_cast<size_t>(m_row) * k + acol,
                                     (m_row < m) && (acol < k));
                    }
                }
                if (!canB16) {
                    #pragma unroll
                    for (int jj = 0; jj < 4; ++jj) {
                        const int ncol = n_col + jj;
                        cp_async_f32(sB2 + jj * 4, B + static_cast<size_t>(b_row) * n + ncol,
                                     (b_row < k) && (ncol < n));
                    }
                }
            }
            asm volatile("cp.async.commit_group;\n" ::);
        }

        // Compute using current buffer while the next buffer loads
        #pragma unroll
        for (int j = 0; j < TILE_SIZE; ++j) {
            acc += As[buf][ty][j] * Bs[buf][j][tx];
        }

        // Before switching to next buffer, ensure the async copies completed
        if (inext < k) {
            asm volatile("cp.async.wait_group 0;\n" ::);
        }
        __syncthreads();
        buf ^= 1;
    }

    if (m_row < m && n_col < n) {
        C[static_cast<size_t>(m_row) * n + n_col] = acc;
    }
#else
    // Fallback: simple shared-memory GEMM if compiled for < SM80
    __shared__ float As[TILE_SIZE][TILE_SIZE + 1];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE + 1];
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int m_row = blockIdx.x * TILE_SIZE + ty;
    const int n_col = blockIdx.y * TILE_SIZE + tx;
    float acc = 0.f;
    for (int k0 = 0; k0 < k; k0 += TILE_SIZE) {
        As[ty][tx] = (m_row < m && (k0 + tx) < k) ? A[static_cast<size_t>(m_row) * k + (k0 + tx)] : 0.f;
        Bs[ty][tx] = (n_col < n && (k0 + ty) < k) ? B[static_cast<size_t>(k0 + ty) * n + n_col] : 0.f;
        __syncthreads();
        #pragma unroll
        for (int j = 0; j < TILE_SIZE; ++j) acc += As[ty][j] * Bs[j][tx];
        __syncthreads();
    }
    if (m_row < m && n_col < n) C[static_cast<size_t>(m_row) * n + n_col] = acc;
#endif
}
