#pragma once

#ifndef TILE_SIZE
#define TILE_SIZE 32
#endif

// Tiled GEMM with shared-memory ping-pong buffers (float4 loads, no async copy)
// Row-major, BLAS-style: C[M,N] = A[M,K] x B[K,N]
__global__ void gemm_shared_pingpong_float4(const float* __restrict__ A,
                                            const float* __restrict__ B,
                                            float* __restrict__ C,
                                            int m, int n, int k) {
    __shared__ float As[2][TILE_SIZE][TILE_SIZE + 8];
    __shared__ float Bs[2][TILE_SIZE][TILE_SIZE + 8];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int m_row = blockIdx.x * TILE_SIZE + ty; // row in C/A (M)
    const int n_col = blockIdx.y * TILE_SIZE + tx; // col in C/B (N)

    float acc = 0.f;

    // Prime first tile (buffer 0)
    int buf = 0;
    int i = 0;
    if (m_row < m) {
        if ((tx & 3) == 0) {
            const int a_col = i + tx;
            if (((k & 3) == 0) && (a_col + 3 < k)) {
                const float4 v = *reinterpret_cast<const float4*>(&A[m_row * k + a_col]);
                *reinterpret_cast<float4*>(&As[buf][ty][tx]) = v;
            } else {
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const int col = a_col + j;
                    As[buf][ty][tx + j] = (col < k) ? A[m_row * k + col] : 0.f;
                }
            }
        }
    } else if ((tx & 3) == 0) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) As[buf][ty][tx + j] = 0.f;
    }

    if ((i + ty) < k) {
        if ((tx & 3) == 0) {
            const int b_col = n_col;
            if (((n & 3) == 0) && (b_col + 3 < n)) {
                const float4 v = *reinterpret_cast<const float4*>(&B[(i + ty) * n + b_col]);
                *reinterpret_cast<float4*>(&Bs[buf][ty][tx]) = v;
            } else {
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const int col = b_col + j;
                    Bs[buf][ty][tx + j] = (col < n) ? B[(i + ty) * n + col] : 0.f;
                }
            }
        }
    } else if ((tx & 3) == 0) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) Bs[buf][ty][tx + j] = 0.f;
    }

    __syncthreads();

    // Main loop over tiles with ping-pong between buffers 0 and 1
    for (i = 0; i < k; i += TILE_SIZE) {
        // Compute on current buffer
        #pragma unroll
        for (int j = 0; j < TILE_SIZE; ++j) {
            acc += As[buf][ty][j] * Bs[buf][j][tx];
        }

        __syncthreads();

        // Preload next tile into the other buffer (no async; structured for clarity)
        const int next = buf ^ 1;
        const int inext = i + TILE_SIZE;
        if (inext < k) {
            if (m_row < m) {
                if ((tx & 3) == 0) {
                    const int a_col = inext + tx;
                    if (((k & 3) == 0) && (a_col + 3 < k)) {
                        const float4 v = *reinterpret_cast<const float4*>(&A[m_row * k + a_col]);
                        *reinterpret_cast<float4*>(&As[next][ty][tx]) = v;
                    } else {
                        #pragma unroll
                        for (int jj = 0; jj < 4; ++jj) {
                            const int col = a_col + jj;
                            As[next][ty][tx + jj] = (col < k) ? A[m_row * k + col] : 0.f;
                        }
                    }
                }
            } else if ((tx & 3) == 0) {
                #pragma unroll
                for (int jj = 0; jj < 4; ++jj) As[next][ty][tx + jj] = 0.f;
            }

            if ((inext + ty) < k) {
                if ((tx & 3) == 0) {
                    const int b_col = n_col;
                    if (((n & 3) == 0) && (b_col + 3 < n)) {
                        const float4 v = *reinterpret_cast<const float4*>(&B[(inext + ty) * n + b_col]);
                        *reinterpret_cast<float4*>(&Bs[next][ty][tx]) = v;
                    } else {
                        #pragma unroll
                        for (int jj = 0; jj < 4; ++jj) {
                            const int col = b_col + jj;
                            Bs[next][ty][tx + jj] = (col < n) ? B[(inext + ty) * n + col] : 0.f;
                        }
                    }
                }
            } else if ((tx & 3) == 0) {
                #pragma unroll
                for (int jj = 0; jj < 4; ++jj) Bs[next][ty][tx + jj] = 0.f;
            }
        }

        __syncthreads();
        buf ^= 1; // switch buffers
    }

    if (m_row < m && n_col < n) {
        C[m_row * n + n_col] = acc;
    }
}

// Tiled (shared-memory) GEMM (row-major, BLAS-style)
// Computes: C[M,N] = A[M,K] x B[K,N]
// Shapes (row-major):
//  - A: M x K
//  - B: K x N
//  - C: M x N
__global__ void gemm_shared_pingpong(const float* __restrict__ A,
                                     const float* __restrict__ B,
                                     float* __restrict__ C,
                                     int m, int n, int k) {
    __shared__ float As[2][TILE_SIZE][TILE_SIZE + 1];
    __shared__ float Bs[2][TILE_SIZE][TILE_SIZE + 1];

    // Block tiles cover n (rows) and k (cols)
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int m_row = blockIdx.x * TILE_SIZE + ty; // row in C/A (M)
    const int n_col = blockIdx.y * TILE_SIZE + tx; // col in C/B (N)

    float acc = 0.f;

    // Prime first tile into buffer 0
    int buf = 0;
    int i = 0;
    if (m_row < m && (i + tx) < k)
        As[buf][ty][tx] = A[m_row * k + (i + tx)];
    else
        As[buf][ty][tx] = 0.f;
    if (n_col < n && (i + ty) < k)
        Bs[buf][ty][tx] = B[(i + ty) * n + n_col];
    else
        Bs[buf][ty][tx] = 0.f;

    __syncthreads();

    // Main loop with ping-pong buffers (no async copy)
    for (i = 0; i < k; i += TILE_SIZE) {
        // Compute using current buffer
        #pragma unroll
        for (int j = 0; j < TILE_SIZE; ++j) acc += As[buf][ty][j] * Bs[buf][j][tx];

        __syncthreads();

        // Load next tile into the other buffer
        const int next = buf ^ 1;
        const int inext = i + TILE_SIZE;
        if (inext < k) {
            if (m_row < m && (inext + tx) < k)
                As[next][ty][tx] = A[m_row * k + (inext + tx)];
            else
                As[next][ty][tx] = 0.f;
            if (n_col < n && (inext + ty) < k)
                Bs[next][ty][tx] = B[(inext + ty) * n + n_col];
            else
                Bs[next][ty][tx] = 0.f;
        }

        __syncthreads();
        buf ^= 1;
    }

    if (m_row < m && n_col < n)
        C[m_row * n + n_col] = acc;
}
