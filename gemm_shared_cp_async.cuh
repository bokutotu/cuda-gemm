#pragma once

#ifndef TILE_SIZE
#define TILE_SIZE 32
#endif

// cp.async-based ping-pong GEMM for SM80+
// Row-major: C[n,k] = A[n,m] x B[m,k]
// Each thread computes one C element; tiles are TILE_SIZE x TILE_SIZE.
// Uses cp.async to overlap global->shared copies for tile t+1 while computing tile t.
__global__ void gemm_shared_cp_async(const float* __restrict__ A,
                                     const float* __restrict__ B,
                                     float* __restrict__ C,
                                     int n, int m, int k) {
#if __CUDA_ARCH__ >= 800
    // Use +8 padding so each row stride is 160B for TILE_SIZE=32,
    // which preserves 16B alignment for cp.async. Also mitigates bank conflicts.
    __shared__ float As[2][TILE_SIZE][TILE_SIZE + 8];
    __shared__ float Bs[2][TILE_SIZE][TILE_SIZE + 8];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int n_addr = blockIdx.x * TILE_SIZE + ty; // row in C/A
    const int k_addr = blockIdx.y * TILE_SIZE + tx; // col in C/B

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
        const int a_col0 = 0 + tx;
        const int b_row0 = 0 + ty;
        uint32_t sA = __cvta_generic_to_shared(&As[buf][ty][tx]);
        uint32_t sB = __cvta_generic_to_shared(&Bs[buf][ty][tx]);
        if ((tx & 3) == 0) {
            // Try 16B vectorized copy when aligned and fully in-bounds
            const bool canA16 = ((m & 3) == 0) && (n_addr < n) && (a_col0 + 3 < m);
            const bool canB16 = ((k & 3) == 0) && (b_row0 < m) && (k_addr + 3 < k);
            const float* gA = A + static_cast<size_t>(n_addr) * m + a_col0;
            const float* gB = B + static_cast<size_t>(b_row0) * k + k_addr;
            cp_async_f16(sA, gA, canA16);
            cp_async_f16(sB, gB, canB16);
            if (!canA16) {
                // Fallback element-wise (covers edges)
                #pragma unroll
                for (int jj = 0; jj < 4; ++jj) {
                    const int acol = a_col0 + jj;
                    cp_async_f32(sA + jj * 4, A + static_cast<size_t>(n_addr) * m + acol,
                                 (n_addr < n) && (acol < m));
                }
            }
            if (!canB16) {
                #pragma unroll
                for (int jj = 0; jj < 4; ++jj) {
                    const int kcol = k_addr + jj;
                    cp_async_f32(sB + jj * 4, B + static_cast<size_t>(b_row0) * k + kcol,
                                 (b_row0 < m) && (kcol < k));
                }
            }
        }
        asm volatile("cp.async.commit_group;\n" ::);
        asm volatile("cp.async.wait_group 0;\n" ::);
        __syncthreads();
    }

    // Main loop over tiles with pipelined copies and compute
    for (int i = 0; i < m; i += TILE_SIZE) {
        const int next = buf ^ 1;
        const int inext = i + TILE_SIZE;

        // Issue async copies for next tile (if any), overlapping with compute below
        if (inext < m) {
            const int a_col = inext + tx;
            const int b_row = inext + ty;

            uint32_t sA2 = __cvta_generic_to_shared(&As[next][ty][tx]);
            uint32_t sB2 = __cvta_generic_to_shared(&Bs[next][ty][tx]);
            if ((tx & 3) == 0) {
                const bool canA16 = ((m & 3) == 0) && (n_addr < n) && (a_col + 3 < m);
                const bool canB16 = ((k & 3) == 0) && (b_row < m) && (k_addr + 3 < k);
                const float* gA = A + static_cast<size_t>(n_addr) * m + a_col;
                const float* gB = B + static_cast<size_t>(b_row) * k + k_addr;
                cp_async_f16(sA2, gA, canA16);
                cp_async_f16(sB2, gB, canB16);
                if (!canA16) {
                    #pragma unroll
                    for (int jj = 0; jj < 4; ++jj) {
                        const int acol = a_col + jj;
                        cp_async_f32(sA2 + jj * 4, A + static_cast<size_t>(n_addr) * m + acol,
                                     (n_addr < n) && (acol < m));
                    }
                }
                if (!canB16) {
                    #pragma unroll
                    for (int jj = 0; jj < 4; ++jj) {
                        const int kcol = k_addr + jj;
                        cp_async_f32(sB2 + jj * 4, B + static_cast<size_t>(b_row) * k + kcol,
                                     (b_row < m) && (kcol < k));
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
        if (inext < m) {
            asm volatile("cp.async.wait_group 0;\n" ::);
        }
        __syncthreads();
        buf ^= 1;
    }

    if (n_addr < n && k_addr < k) {
        C[static_cast<size_t>(n_addr) * k + k_addr] = acc;
    }
#else
    // Fallback: simple shared-memory GEMM if compiled for < SM80
    __shared__ float As[TILE_SIZE][TILE_SIZE + 1];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE + 1];
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int n_addr = blockIdx.x * TILE_SIZE + ty;
    const int k_addr = blockIdx.y * TILE_SIZE + tx;
    float acc = 0.f;
    for (int i = 0; i < m; i += TILE_SIZE) {
        As[ty][tx] = (n_addr < n && (i + tx) < m) ? A[static_cast<size_t>(n_addr) * m + (i + tx)] : 0.f;
        Bs[ty][tx] = (k_addr < k && (i + ty) < m) ? B[static_cast<size_t>(i + ty) * k + k_addr] : 0.f;
        __syncthreads();
        #pragma unroll
        for (int j = 0; j < TILE_SIZE; ++j) acc += As[ty][j] * Bs[j][tx];
        __syncthreads();
    }
    if (n_addr < n && k_addr < k) C[static_cast<size_t>(n_addr) * k + k_addr] = acc;
#endif
}

