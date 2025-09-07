#pragma once

#ifndef TILE_SIZE
#define TILE_SIZE 32
#endif

// Thread-level register tiling sizes (per-thread micro-tile)
#ifndef REG_TILE_M
#define REG_TILE_M 4
#endif
#ifndef REG_TILE_N
#define REG_TILE_N 4
#endif

static_assert(TILE_SIZE % REG_TILE_M == 0, "TILE_SIZE must be divisible by REG_TILE_M");
static_assert(TILE_SIZE % REG_TILE_N == 0, "TILE_SIZE must be divisible by REG_TILE_N");

// Launch geometry (recommended):
//   dim3 block(TILE_SIZE/REG_TILE_N, TILE_SIZE/REG_TILE_M);
//   dim3 grid((n + TILE_SIZE - 1) / TILE_SIZE,
//             (k + TILE_SIZE - 1) / TILE_SIZE);
// Each thread computes a REG_TILE_M x REG_TILE_N micro-tile entirely in registers.

// ---------------------------------------------
// Shared (scalar global loads) + register tile
// ---------------------------------------------
__global__ void gemm_shared_register_tile(const float* __restrict__ A,
                                          const float* __restrict__ B,
                                          float* __restrict__ C,
                                          int n, int m, int k) {
    __shared__ float As[TILE_SIZE][TILE_SIZE + 1];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE + 1];

    const int tx = threadIdx.x; // 0 .. TILE_SIZE/REG_TILE_N-1
    const int ty = threadIdx.y; // 0 .. TILE_SIZE/REG_TILE_M-1

    const int tile_row0 = blockIdx.x * TILE_SIZE; // base row of C tile
    const int tile_col0 = blockIdx.y * TILE_SIZE; // base col of C tile

    const int out_row0 = tile_row0 + ty * REG_TILE_M; // this thread's sub-tile origin (row)
    const int out_col0 = tile_col0 + tx * REG_TILE_N; // this thread's sub-tile origin (col)

    float acc[REG_TILE_M][REG_TILE_N];
    #pragma unroll
    for (int i = 0; i < REG_TILE_M; ++i) {
        #pragma unroll
        for (int j = 0; j < REG_TILE_N; ++j) acc[i][j] = 0.f;
    }

    for (int t = 0; t < m; t += TILE_SIZE) {
        // Load A and B tiles to shared. Each thread loads REG_TILE_M x REG_TILE_N elements.
        #pragma unroll
        for (int ii = 0; ii < REG_TILE_M; ++ii) {
            const int a_row = ty * REG_TILE_M + ii;          // 0..TILE_SIZE-1
            const int ga_row = tile_row0 + a_row;            // global A row
            const int b_row = a_row;                         // B tile row (same index within tile)
            const int gb_row = t + b_row;                    // global B row

            #pragma unroll
            for (int jj = 0; jj < REG_TILE_N; ++jj) {
                const int a_col = tx * REG_TILE_N + jj;      // 0..TILE_SIZE-1
                const int ga_col = t + a_col;                // global A col
                As[a_row][a_col] = (ga_row < n && ga_col < m)
                                    ? A[static_cast<size_t>(ga_row) * m + ga_col]
                                    : 0.f;

                const int b_col = tx * REG_TILE_N + jj;      // 0..TILE_SIZE-1
                const int gb_col = tile_col0 + b_col;        // global B col
                Bs[b_row][b_col] = (gb_row < m && gb_col < k)
                                    ? B[static_cast<size_t>(gb_row) * k + gb_col]
                                    : 0.f;
            }
        }

        __syncthreads();

        // Compute: iterate K-tile dimension inside shared tiles
        #pragma unroll
        for (int p = 0; p < TILE_SIZE; ++p) {
            float a_frag[REG_TILE_M];
            float b_frag[REG_TILE_N];
            #pragma unroll
            for (int ii = 0; ii < REG_TILE_M; ++ii) {
                a_frag[ii] = As[ty * REG_TILE_M + ii][p];
            }
            #pragma unroll
            for (int jj = 0; jj < REG_TILE_N; ++jj) {
                b_frag[jj] = Bs[p][tx * REG_TILE_N + jj];
            }
            #pragma unroll
            for (int ii = 0; ii < REG_TILE_M; ++ii) {
                #pragma unroll
                for (int jj = 0; jj < REG_TILE_N; ++jj) {
                    acc[ii][jj] += a_frag[ii] * b_frag[jj];
                }
            }
        }

        __syncthreads();
    }

    // Write results
    #pragma unroll
    for (int ii = 0; ii < REG_TILE_M; ++ii) {
        const int gr = out_row0 + ii;
        if (gr < n) {
            #pragma unroll
            for (int jj = 0; jj < REG_TILE_N; ++jj) {
                const int gc = out_col0 + jj;
                if (gc < k) C[static_cast<size_t>(gr) * k + gc] = acc[ii][jj];
            }
        }
    }
}

// -------------------------------------------------
// cp.async ping-pong (SM80+) + register tile
// Each thread computes REG_TILE_M x REG_TILE_N in registers while
// overlapping global->shared copies of the next tile using cp.async.
// Fallback path for < SM80 mirrors the scalar shared+register implementation.
// -------------------------------------------------
__global__ void gemm_shared_cp_async_register_tile(const float* __restrict__ A,
                                                   const float* __restrict__ B,
                                                   float* __restrict__ C,
                                                   int n, int m, int k) {
#if __CUDA_ARCH__ >= 800
    __shared__ float As[2][TILE_SIZE][TILE_SIZE + 8];
    __shared__ float Bs[2][TILE_SIZE][TILE_SIZE + 8];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tile_row0 = blockIdx.x * TILE_SIZE;
    const int tile_col0 = blockIdx.y * TILE_SIZE;
    const int out_row0  = tile_row0 + ty * REG_TILE_M;
    const int out_col0  = tile_col0 + tx * REG_TILE_N;

    float acc[REG_TILE_M][REG_TILE_N];
    #pragma unroll
    for (int i = 0; i < REG_TILE_M; ++i) {
        #pragma unroll
        for (int j = 0; j < REG_TILE_N; ++j) acc[i][j] = 0.f;
    }

    auto cp_async_f32 = [](uint32_t dst_smem, const float* src_gmem, bool pred) {
        if (pred) {
            asm volatile("cp.async.ca.shared.global [%0], [%1], %2;\n" ::
                         "r"(dst_smem), "l"(src_gmem), "n"(4));
        } else {
            asm volatile("st.shared.f32 [%0], %1;\n" :: "r"(dst_smem), "f"(0.0f));
        }
    };
    auto cp_async_f16 = [](uint32_t dst_smem, const float* src_gmem, bool pred) {
        if (pred) {
            asm volatile("cp.async.ca.shared.global [%0], [%1], %2;\n" ::
                         "r"(dst_smem), "l"(src_gmem), "n"(16));
        } else {
            // Zero-fill 16B (4 floats)
            asm volatile("st.shared.f32 [%0], %1;\n" :: "r"(dst_smem + 0),  "f"(0.0f));
            asm volatile("st.shared.f32 [%0], %1;\n" :: "r"(dst_smem + 4),  "f"(0.0f));
            asm volatile("st.shared.f32 [%0], %1;\n" :: "r"(dst_smem + 8),  "f"(0.0f));
            asm volatile("st.shared.f32 [%0], %1;\n" :: "r"(dst_smem + 12), "f"(0.0f));
        }
    };

    int buf = 0;

    // Prime tile 0 into buffer 0
    {
        const int tbase = 0;
        #pragma unroll
        for (int ii = 0; ii < REG_TILE_M; ++ii) {
            const int a_row = ty * REG_TILE_M + ii;
            const int ga_row = tile_row0 + a_row;
            const int b_row = a_row;
            const int gb_row = tbase + b_row;
            const int a_col0 = tx * REG_TILE_N;
            const int b_col0 = tx * REG_TILE_N;

            if ((REG_TILE_N % 4) == 0) {
                #pragma unroll
                for (int v = 0; v < REG_TILE_N; v += 4) {
                    uint32_t sA = __cvta_generic_to_shared(&As[buf][a_row][a_col0 + v]);
                    uint32_t sB = __cvta_generic_to_shared(&Bs[buf][b_row][b_col0 + v]);
                    const bool canA16 = ((m & 3) == 0) && (ga_row < n) && ((tbase + a_col0 + v + 3) < m);
                    const bool canB16 = ((k & 3) == 0) && (gb_row < m) && ((tile_col0 + b_col0 + v + 3) < k);
                    const float* gA = A + static_cast<size_t>(ga_row) * m + (tbase + a_col0 + v);
                    const float* gB = B + static_cast<size_t>(gb_row) * k + (tile_col0 + b_col0 + v);
                    cp_async_f16(sA, gA, canA16);
                    cp_async_f16(sB, gB, canB16);
                    if (!canA16) {
                        #pragma unroll
                        for (int jj = 0; jj < 4; ++jj)
                            cp_async_f32(sA + jj * 4, gA + jj, (ga_row < n) && ((tbase + a_col0 + v + jj) < m));
                    }
                    if (!canB16) {
                        #pragma unroll
                        for (int jj = 0; jj < 4; ++jj)
                            cp_async_f32(sB + jj * 4, gB + jj, (gb_row < m) && ((tile_col0 + b_col0 + v + jj) < k));
                    }
                }
            } else {
                #pragma unroll
                for (int jj = 0; jj < REG_TILE_N; ++jj) {
                    const int ga_col = tbase + a_col0 + jj;
                    uint32_t sA = __cvta_generic_to_shared(&As[buf][a_row][a_col0 + jj]);
                    const float* gA = A + static_cast<size_t>(ga_row) * m + ga_col;
                    cp_async_f32(sA, gA, (ga_row < n) && (ga_col < m));
                    const int gb_col = tile_col0 + b_col0 + jj;
                    uint32_t sB = __cvta_generic_to_shared(&Bs[buf][b_row][b_col0 + jj]);
                    const float* gB = B + static_cast<size_t>(gb_row) * k + gb_col;
                    cp_async_f32(sB, gB, (gb_row < m) && (gb_col < k));
                }
            }
        }
        asm volatile("cp.async.commit_group;\n" ::);
        asm volatile("cp.async.wait_group 0;\n" ::);
        __syncthreads();
    }

    for (int t = 0; t < m; t += TILE_SIZE) {
        const int next = buf ^ 1;
        const int inext = t + TILE_SIZE;

        // Preload next tile while computing current
        if (inext < m) {
            #pragma unroll
            for (int ii = 0; ii < REG_TILE_M; ++ii) {
                const int a_row = ty * REG_TILE_M + ii;
                const int ga_row = tile_row0 + a_row;
                const int b_row = a_row;
                const int gb_row = inext + b_row;
                const int a_col0 = tx * REG_TILE_N;
                const int b_col0 = tx * REG_TILE_N;

                if ((REG_TILE_N % 4) == 0) {
                    #pragma unroll
                    for (int v = 0; v < REG_TILE_N; v += 4) {
                        uint32_t sA = __cvta_generic_to_shared(&As[next][a_row][a_col0 + v]);
                        uint32_t sB = __cvta_generic_to_shared(&Bs[next][b_row][b_col0 + v]);
                        const bool canA16 = ((m & 3) == 0) && (ga_row < n) && ((inext + a_col0 + v + 3) < m);
                        const bool canB16 = ((k & 3) == 0) && (gb_row < m) && ((tile_col0 + b_col0 + v + 3) < k);
                        const float* gA = A + static_cast<size_t>(ga_row) * m + (inext + a_col0 + v);
                        const float* gB = B + static_cast<size_t>(gb_row) * k + (tile_col0 + b_col0 + v);
                        cp_async_f16(sA, gA, canA16);
                        cp_async_f16(sB, gB, canB16);
                        if (!canA16) {
                            #pragma unroll
                            for (int jj = 0; jj < 4; ++jj)
                                cp_async_f32(sA + jj * 4, gA + jj, (ga_row < n) && ((inext + a_col0 + v + jj) < m));
                        }
                        if (!canB16) {
                            #pragma unroll
                            for (int jj = 0; jj < 4; ++jj)
                                cp_async_f32(sB + jj * 4, gB + jj, (gb_row < m) && ((tile_col0 + b_col0 + v + jj) < k));
                        }
                    }
                } else {
                    #pragma unroll
                    for (int jj = 0; jj < REG_TILE_N; ++jj) {
                        const int ga_col = inext + a_col0 + jj;
                        uint32_t sA = __cvta_generic_to_shared(&As[next][a_row][a_col0 + jj]);
                        const float* gA = A + static_cast<size_t>(ga_row) * m + ga_col;
                        cp_async_f32(sA, gA, (ga_row < n) && (ga_col < m));
                        const int gb_col = tile_col0 + b_col0 + jj;
                        uint32_t sB = __cvta_generic_to_shared(&Bs[next][b_row][b_col0 + jj]);
                        const float* gB = B + static_cast<size_t>(gb_row) * k + gb_col;
                        cp_async_f32(sB, gB, (gb_row < m) && (gb_col < k));
                    }
                }
            }
            asm volatile("cp.async.commit_group;\n" ::);
        }

        // Compute on current buffer
        #pragma unroll
        for (int p = 0; p < TILE_SIZE; ++p) {
            float a_frag[REG_TILE_M];
            float b_frag[REG_TILE_N];
            #pragma unroll
            for (int ii = 0; ii < REG_TILE_M; ++ii) a_frag[ii] = As[buf][ty * REG_TILE_M + ii][p];
            #pragma unroll
            for (int jj = 0; jj < REG_TILE_N; ++jj) b_frag[jj] = Bs[buf][p][tx * REG_TILE_N + jj];
            #pragma unroll
            for (int ii = 0; ii < REG_TILE_M; ++ii) {
                #pragma unroll
                for (int jj = 0; jj < REG_TILE_N; ++jj) acc[ii][jj] += a_frag[ii] * b_frag[jj];
            }
        }

        if (inext < m) {
            asm volatile("cp.async.wait_group 0;\n" ::);
        }
        __syncthreads();
        buf ^= 1;
    }

    // Store results
    #pragma unroll
    for (int ii = 0; ii < REG_TILE_M; ++ii) {
        const int gr = out_row0 + ii;
        if (gr < n) {
            #pragma unroll
            for (int jj = 0; jj < REG_TILE_N; ++jj) {
                const int gc = out_col0 + jj;
                if (gc < k) C[static_cast<size_t>(gr) * k + gc] = acc[ii][jj];
            }
        }
    }
#else
    // Fallback: shared + register tile (no async, no ping-pong)
    __shared__ float As[TILE_SIZE][TILE_SIZE + 1];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE + 1];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tile_row0 = blockIdx.x * TILE_SIZE;
    const int tile_col0 = blockIdx.y * TILE_SIZE;
    const int out_row0  = tile_row0 + ty * REG_TILE_M;
    const int out_col0  = tile_col0 + tx * REG_TILE_N;

    float acc[REG_TILE_M][REG_TILE_N];
    #pragma unroll
    for (int i = 0; i < REG_TILE_M; ++i) {
        #pragma unroll
        for (int j = 0; j < REG_TILE_N; ++j) acc[i][j] = 0.f;
    }

    for (int t = 0; t < m; t += TILE_SIZE) {
        #pragma unroll
        for (int ii = 0; ii < REG_TILE_M; ++ii) {
            const int a_row = ty * REG_TILE_M + ii;
            const int ga_row = tile_row0 + a_row;
            const int b_row = a_row;
            const int gb_row = t + b_row;
            #pragma unroll
            for (int jj = 0; jj < REG_TILE_N; ++jj) {
                const int a_col = tx * REG_TILE_N + jj;
                const int ga_col = t + a_col;
                As[a_row][a_col] = (ga_row < n && ga_col < m) ? A[static_cast<size_t>(ga_row) * m + ga_col] : 0.f;
                const int b_col = tx * REG_TILE_N + jj;
                const int gb_col = tile_col0 + b_col;
                Bs[b_row][b_col] = (gb_row < m && gb_col < k) ? B[static_cast<size_t>(gb_row) * k + gb_col] : 0.f;
            }
        }
        __syncthreads();
        #pragma unroll
        for (int p = 0; p < TILE_SIZE; ++p) {
            float a_frag[REG_TILE_M];
            float b_frag[REG_TILE_N];
            #pragma unroll
            for (int ii = 0; ii < REG_TILE_M; ++ii) a_frag[ii] = As[ty * REG_TILE_M + ii][p];
            #pragma unroll
            for (int jj = 0; jj < REG_TILE_N; ++jj) b_frag[jj] = Bs[p][tx * REG_TILE_N + jj];
            #pragma unroll
            for (int ii = 0; ii < REG_TILE_M; ++ii)
                #pragma unroll
                for (int jj = 0; jj < REG_TILE_N; ++jj) acc[ii][jj] += a_frag[ii] * b_frag[jj];
        }
        __syncthreads();
    }

    #pragma unroll
    for (int ii = 0; ii < REG_TILE_M; ++ii) {
        const int gr = out_row0 + ii;
        if (gr < n) {
            #pragma unroll
            for (int jj = 0; jj < REG_TILE_N; ++jj) {
                const int gc = out_col0 + jj;
                if (gc < k) C[static_cast<size_t>(gr) * k + gc] = acc[ii][jj];
            }
        }
    }
#endif
}

// -------------------------------------------------
// Shared (float4 global loads) + register tile
// -------------------------------------------------
__global__ void gemm_shared_register_tile_float4(const float* __restrict__ A,
                                                 const float* __restrict__ B,
                                                 float* __restrict__ C,
                                                 int n, int m, int k) {
    __shared__ float As[TILE_SIZE][TILE_SIZE + 8];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE + 8];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tile_row0 = blockIdx.x * TILE_SIZE;
    const int tile_col0 = blockIdx.y * TILE_SIZE;
    const int out_row0 = tile_row0 + ty * REG_TILE_M;
    const int out_col0 = tile_col0 + tx * REG_TILE_N;

    float acc[REG_TILE_M][REG_TILE_N];
    #pragma unroll
    for (int i = 0; i < REG_TILE_M; ++i) {
        #pragma unroll
        for (int j = 0; j < REG_TILE_N; ++j) acc[i][j] = 0.f;
    }

    for (int t = 0; t < m; t += TILE_SIZE) {
        // Load tile for A and B; vectorize along N-dim (columns) where possible.
        #pragma unroll
        for (int ii = 0; ii < REG_TILE_M; ++ii) {
            const int a_row = ty * REG_TILE_M + ii;
            const int ga_row = tile_row0 + a_row;
            const int b_row = a_row;
            const int gb_row = t + b_row;

            const int a_col0 = tx * REG_TILE_N;
            const int b_col0 = tx * REG_TILE_N;

            if ((REG_TILE_N % 4) == 0) {
                #pragma unroll
                for (int v = 0; v < REG_TILE_N; v += 4) {
                    const bool canA16 = ((m & 3) == 0) && (ga_row < n) && ((t + a_col0 + v + 3) < m);
                    const bool canB16 = ((k & 3) == 0) && (gb_row < m) && ((tile_col0 + b_col0 + v + 3) < k);
                    if (canA16) {
                        const float4 va = *reinterpret_cast<const float4*>(&A[static_cast<size_t>(ga_row) * m + (t + a_col0 + v)]);
                        *reinterpret_cast<float4*>(&As[a_row][a_col0 + v]) = va;
                    } else {
                        #pragma unroll
                        for (int jj = 0; jj < 4; ++jj) {
                            const int ga_col = t + a_col0 + v + jj;
                            As[a_row][a_col0 + v + jj] = (ga_row < n && ga_col < m)
                                ? A[static_cast<size_t>(ga_row) * m + ga_col] : 0.f;
                        }
                    }
                    if (canB16) {
                        const float4 vb = *reinterpret_cast<const float4*>(&B[static_cast<size_t>(gb_row) * k + (tile_col0 + b_col0 + v)]);
                        *reinterpret_cast<float4*>(&Bs[b_row][b_col0 + v]) = vb;
                    } else {
                        #pragma unroll
                        for (int jj = 0; jj < 4; ++jj) {
                            const int gb_col = tile_col0 + b_col0 + v + jj;
                            Bs[b_row][b_col0 + v + jj] = (gb_row < m && gb_col < k)
                                ? B[static_cast<size_t>(gb_row) * k + gb_col] : 0.f;
                        }
                    }
                }
            } else {
                #pragma unroll
                for (int jj = 0; jj < REG_TILE_N; ++jj) {
                    const int ga_col = t + a_col0 + jj;
                    As[a_row][a_col0 + jj] = (ga_row < n && ga_col < m)
                        ? A[static_cast<size_t>(ga_row) * m + ga_col] : 0.f;
                }
                #pragma unroll
                for (int jj = 0; jj < REG_TILE_N; ++jj) {
                    const int gb_col = tile_col0 + b_col0 + jj;
                    Bs[b_row][b_col0 + jj] = (gb_row < m && gb_col < k)
                        ? B[static_cast<size_t>(gb_row) * k + gb_col] : 0.f;
                }
            }
        }

        __syncthreads();

        #pragma unroll
        for (int p = 0; p < TILE_SIZE; ++p) {
            float a_frag[REG_TILE_M];
            float b_frag[REG_TILE_N];
            #pragma unroll
            for (int ii = 0; ii < REG_TILE_M; ++ii) a_frag[ii] = As[ty * REG_TILE_M + ii][p];
            #pragma unroll
            for (int jj = 0; jj < REG_TILE_N; ++jj) b_frag[jj] = Bs[p][tx * REG_TILE_N + jj];
            #pragma unroll
            for (int ii = 0; ii < REG_TILE_M; ++ii) {
                #pragma unroll
                for (int jj = 0; jj < REG_TILE_N; ++jj) acc[ii][jj] += a_frag[ii] * b_frag[jj];
            }
        }

        __syncthreads();
    }

    #pragma unroll
    for (int ii = 0; ii < REG_TILE_M; ++ii) {
        const int gr = out_row0 + ii;
        if (gr < n) {
            #pragma unroll
            for (int jj = 0; jj < REG_TILE_N; ++jj) {
                const int gc = out_col0 + jj;
                if (gc < k) C[static_cast<size_t>(gr) * k + gc] = acc[ii][jj];
            }
        }
    }
}

// -------------------------------------------------
// Ping-pong (scalar loads) + register tile
// -------------------------------------------------
__global__ void gemm_shared_register_tile_pingpong(const float* __restrict__ A,
                                                   const float* __restrict__ B,
                                                   float* __restrict__ C,
                                                   int n, int m, int k) {
    __shared__ float As[2][TILE_SIZE][TILE_SIZE + 1];
    __shared__ float Bs[2][TILE_SIZE][TILE_SIZE + 1];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tile_row0 = blockIdx.x * TILE_SIZE;
    const int tile_col0 = blockIdx.y * TILE_SIZE;
    const int out_row0 = tile_row0 + ty * REG_TILE_M;
    const int out_col0 = tile_col0 + tx * REG_TILE_N;

    float acc[REG_TILE_M][REG_TILE_N];
    #pragma unroll
    for (int i = 0; i < REG_TILE_M; ++i) {
        #pragma unroll
        for (int j = 0; j < REG_TILE_N; ++j) acc[i][j] = 0.f;
    }

    int buf = 0;

    // Prime buffer 0
    {
        const int t = 0;
        #pragma unroll
        for (int ii = 0; ii < REG_TILE_M; ++ii) {
            const int a_row = ty * REG_TILE_M + ii;
            const int ga_row = tile_row0 + a_row;
            const int b_row = a_row;
            const int gb_row = t + b_row;
            #pragma unroll
            for (int jj = 0; jj < REG_TILE_N; ++jj) {
                const int a_col = tx * REG_TILE_N + jj;
                const int ga_col = t + a_col;
                As[buf][a_row][a_col] = (ga_row < n && ga_col < m)
                    ? A[static_cast<size_t>(ga_row) * m + ga_col] : 0.f;
                const int b_col = tx * REG_TILE_N + jj;
                const int gb_col = tile_col0 + b_col;
                Bs[buf][b_row][b_col] = (gb_row < m && gb_col < k)
                    ? B[static_cast<size_t>(gb_row) * k + gb_col] : 0.f;
            }
        }
        __syncthreads();
    }

    for (int t = 0; t < m; t += TILE_SIZE) {
        // Compute on current buffer
        #pragma unroll
        for (int p = 0; p < TILE_SIZE; ++p) {
            float a_frag[REG_TILE_M];
            float b_frag[REG_TILE_N];
            #pragma unroll
            for (int ii = 0; ii < REG_TILE_M; ++ii) a_frag[ii] = As[buf][ty * REG_TILE_M + ii][p];
            #pragma unroll
            for (int jj = 0; jj < REG_TILE_N; ++jj) b_frag[jj] = Bs[buf][p][tx * REG_TILE_N + jj];
            #pragma unroll
            for (int ii = 0; ii < REG_TILE_M; ++ii) {
                #pragma unroll
                for (int jj = 0; jj < REG_TILE_N; ++jj) acc[ii][jj] += a_frag[ii] * b_frag[jj];
            }
        }

        __syncthreads();

        const int next = buf ^ 1;
        const int tn = t + TILE_SIZE;
        if (tn < m) {
            // Preload next tile into the other buffer
            #pragma unroll
            for (int ii = 0; ii < REG_TILE_M; ++ii) {
                const int a_row = ty * REG_TILE_M + ii;
                const int ga_row = tile_row0 + a_row;
                const int b_row = a_row;
                const int gb_row = tn + b_row;
                #pragma unroll
                for (int jj = 0; jj < REG_TILE_N; ++jj) {
                    const int a_col = tx * REG_TILE_N + jj;
                    const int ga_col = tn + a_col;
                    As[next][a_row][a_col] = (ga_row < n && ga_col < m)
                        ? A[static_cast<size_t>(ga_row) * m + ga_col] : 0.f;
                    const int b_col = tx * REG_TILE_N + jj;
                    const int gb_col = tile_col0 + b_col;
                    Bs[next][b_row][b_col] = (gb_row < m && gb_col < k)
                        ? B[static_cast<size_t>(gb_row) * k + gb_col] : 0.f;
                }
            }
        }

        __syncthreads();
        buf ^= 1;
    }

    #pragma unroll
    for (int ii = 0; ii < REG_TILE_M; ++ii) {
        const int gr = out_row0 + ii;
        if (gr < n) {
            #pragma unroll
            for (int jj = 0; jj < REG_TILE_N; ++jj) {
                const int gc = out_col0 + jj;
                if (gc < k) C[static_cast<size_t>(gr) * k + gc] = acc[ii][jj];
            }
        }
    }
}

// -------------------------------------------------
// Ping-pong (float4 loads) + register tile
// -------------------------------------------------
__global__ void gemm_shared_register_tile_pingpong_float4(const float* __restrict__ A,
                                                          const float* __restrict__ B,
                                                          float* __restrict__ C,
                                                          int n, int m, int k) {
    __shared__ float As[2][TILE_SIZE][TILE_SIZE + 8];
    __shared__ float Bs[2][TILE_SIZE][TILE_SIZE + 8];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tile_row0 = blockIdx.x * TILE_SIZE;
    const int tile_col0 = blockIdx.y * TILE_SIZE;
    const int out_row0 = tile_row0 + ty * REG_TILE_M;
    const int out_col0 = tile_col0 + tx * REG_TILE_N;

    float acc[REG_TILE_M][REG_TILE_N];
    #pragma unroll
    for (int i = 0; i < REG_TILE_M; ++i) {
        #pragma unroll
        for (int j = 0; j < REG_TILE_N; ++j) acc[i][j] = 0.f;
    }

    int buf = 0;

    auto load_tile_float4 = [&](int which, int t_base) {
        #pragma unroll
        for (int ii = 0; ii < REG_TILE_M; ++ii) {
            const int a_row = ty * REG_TILE_M + ii;
            const int ga_row = tile_row0 + a_row;
            const int b_row = a_row;
            const int gb_row = t_base + b_row;
            const int a_col0 = tx * REG_TILE_N;
            const int b_col0 = tx * REG_TILE_N;

            if ((REG_TILE_N % 4) == 0) {
                #pragma unroll
                for (int v = 0; v < REG_TILE_N; v += 4) {
                    const bool canA16 = ((m & 3) == 0) && (ga_row < n) && ((t_base + a_col0 + v + 3) < m);
                    const bool canB16 = ((k & 3) == 0) && (gb_row < m) && ((tile_col0 + b_col0 + v + 3) < k);
                    if (canA16) {
                        const float4 va = *reinterpret_cast<const float4*>(&A[static_cast<size_t>(ga_row) * m + (t_base + a_col0 + v)]);
                        *reinterpret_cast<float4*>(&As[which][a_row][a_col0 + v]) = va;
                    } else {
                        #pragma unroll
                        for (int jj = 0; jj < 4; ++jj) {
                            const int ga_col = t_base + a_col0 + v + jj;
                            As[which][a_row][a_col0 + v + jj] = (ga_row < n && ga_col < m)
                                ? A[static_cast<size_t>(ga_row) * m + ga_col] : 0.f;
                        }
                    }
                    if (canB16) {
                        const float4 vb = *reinterpret_cast<const float4*>(&B[static_cast<size_t>(gb_row) * k + (tile_col0 + b_col0 + v)]);
                        *reinterpret_cast<float4*>(&Bs[which][b_row][b_col0 + v]) = vb;
                    } else {
                        #pragma unroll
                        for (int jj = 0; jj < 4; ++jj) {
                            const int gb_col = tile_col0 + b_col0 + v + jj;
                            Bs[which][b_row][b_col0 + v + jj] = (gb_row < m && gb_col < k)
                                ? B[static_cast<size_t>(gb_row) * k + gb_col] : 0.f;
                        }
                    }
                }
            } else {
                #pragma unroll
                for (int jj = 0; jj < REG_TILE_N; ++jj) {
                    const int ga_col = t_base + a_col0 + jj;
                    As[which][a_row][a_col0 + jj] = (ga_row < n && ga_col < m)
                        ? A[static_cast<size_t>(ga_row) * m + ga_col] : 0.f;
                }
                #pragma unroll
                for (int jj = 0; jj < REG_TILE_N; ++jj) {
                    const int gb_col = tile_col0 + b_col0 + jj;
                    Bs[which][b_row][b_col0 + jj] = (gb_row < m && gb_col < k)
                        ? B[static_cast<size_t>(gb_row) * k + gb_col] : 0.f;
                }
            }
        }
    };

    // Prime buffer 0
    load_tile_float4(/*which=*/buf, /*t_base=*/0);
    __syncthreads();

    for (int t = 0; t < m; t += TILE_SIZE) {
        // Compute on current buffer
        #pragma unroll
        for (int p = 0; p < TILE_SIZE; ++p) {
            float a_frag[REG_TILE_M];
            float b_frag[REG_TILE_N];
            #pragma unroll
            for (int ii = 0; ii < REG_TILE_M; ++ii) a_frag[ii] = As[buf][ty * REG_TILE_M + ii][p];
            #pragma unroll
            for (int jj = 0; jj < REG_TILE_N; ++jj) b_frag[jj] = Bs[buf][p][tx * REG_TILE_N + jj];
            #pragma unroll
            for (int ii = 0; ii < REG_TILE_M; ++ii) {
                #pragma unroll
                for (int jj = 0; jj < REG_TILE_N; ++jj) acc[ii][jj] += a_frag[ii] * b_frag[jj];
            }
        }

        __syncthreads();

        const int next = buf ^ 1;
        const int tn = t + TILE_SIZE;
        if (tn < m) load_tile_float4(/*which=*/next, /*t_base=*/tn);

        __syncthreads();
        buf ^= 1;
    }

    #pragma unroll
    for (int ii = 0; ii < REG_TILE_M; ++ii) {
        const int gr = out_row0 + ii;
        if (gr < n) {
            #pragma unroll
            for (int jj = 0; jj < REG_TILE_N; ++jj) {
                const int gc = out_col0 + jj;
                if (gc < k) C[static_cast<size_t>(gr) * k + gc] = acc[ii][jj];
            }
        }
    }
}
