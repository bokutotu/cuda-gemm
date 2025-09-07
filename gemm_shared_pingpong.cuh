#pragma once

#ifndef TILE_SIZE
#define TILE_SIZE 32
#endif

// Tiled GEMM with shared-memory ping-pong buffers (float4 loads, no async copy)
// Row-major: C[n,k] = A[n,m] x B[m,k]
__global__ void gemm_shared_pingpong_float4(const float* __restrict__ A,
                                            const float* __restrict__ B,
                                            float* __restrict__ C,
                                            int n, int m, int k) {
    __shared__ float As[2][TILE_SIZE][TILE_SIZE + 8];
    __shared__ float Bs[2][TILE_SIZE][TILE_SIZE + 8];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int n_addr = blockIdx.x * TILE_SIZE + ty; // row in C/A
    const int k_addr = blockIdx.y * TILE_SIZE + tx; // col in C/B

    float acc = 0.f;

    // Prime first tile (buffer 0)
    int buf = 0;
    int i = 0;
    if (n_addr < n) {
        if ((tx & 3) == 0) {
            const int a_col = i + tx;
            if (((m & 3) == 0) && (a_col + 3 < m)) {
                const float4 v = *reinterpret_cast<const float4*>(&A[n_addr * m + a_col]);
                *reinterpret_cast<float4*>(&As[buf][ty][tx]) = v;
            } else {
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const int col = a_col + j;
                    As[buf][ty][tx + j] = (col < m) ? A[n_addr * m + col] : 0.f;
                }
            }
        }
    } else if ((tx & 3) == 0) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) As[buf][ty][tx + j] = 0.f;
    }

    if ((i + ty) < m) {
        if ((tx & 3) == 0) {
            const int b_col = k_addr;
            if (((k & 3) == 0) && (b_col + 3 < k)) {
                const float4 v = *reinterpret_cast<const float4*>(&B[(i + ty) * k + b_col]);
                *reinterpret_cast<float4*>(&Bs[buf][ty][tx]) = v;
            } else {
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const int col = b_col + j;
                    Bs[buf][ty][tx + j] = (col < k) ? B[(i + ty) * k + col] : 0.f;
                }
            }
        }
    } else if ((tx & 3) == 0) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) Bs[buf][ty][tx + j] = 0.f;
    }

    __syncthreads();

    // Main loop over tiles with ping-pong between buffers 0 and 1
    for (i = 0; i < m; i += TILE_SIZE) {
        // Compute on current buffer
        #pragma unroll
        for (int j = 0; j < TILE_SIZE; ++j) {
            acc += As[buf][ty][j] * Bs[buf][j][tx];
        }

        __syncthreads();

        // Preload next tile into the other buffer (no async; structured for clarity)
        const int next = buf ^ 1;
        const int inext = i + TILE_SIZE;
        if (inext < m) {
            if (n_addr < n) {
                if ((tx & 3) == 0) {
                    const int a_col = inext + tx;
                    if (((m & 3) == 0) && (a_col + 3 < m)) {
                        const float4 v = *reinterpret_cast<const float4*>(&A[n_addr * m + a_col]);
                        *reinterpret_cast<float4*>(&As[next][ty][tx]) = v;
                    } else {
                        #pragma unroll
                        for (int jj = 0; jj < 4; ++jj) {
                            const int col = a_col + jj;
                            As[next][ty][tx + jj] = (col < m) ? A[n_addr * m + col] : 0.f;
                        }
                    }
                }
            } else if ((tx & 3) == 0) {
                #pragma unroll
                for (int jj = 0; jj < 4; ++jj) As[next][ty][tx + jj] = 0.f;
            }

            if ((inext + ty) < m) {
                if ((tx & 3) == 0) {
                    const int b_col = k_addr;
                    if (((k & 3) == 0) && (b_col + 3 < k)) {
                        const float4 v = *reinterpret_cast<const float4*>(&B[(inext + ty) * k + b_col]);
                        *reinterpret_cast<float4*>(&Bs[next][ty][tx]) = v;
                    } else {
                        #pragma unroll
                        for (int jj = 0; jj < 4; ++jj) {
                            const int col = b_col + jj;
                            Bs[next][ty][tx + jj] = (col < k) ? B[(inext + ty) * k + col] : 0.f;
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

    if (n_addr < n && k_addr < k) {
        C[n_addr * k + k_addr] = acc;
    }
}

// Tiled (shared-memory) matrix multiplication (row-major)
// Computes: C[n,k] = A[n,m] x B[m,k]
// Shapes (row-major):
//  - A: n x m
//  - B: m x k
//  - C: n x k
__global__ void gemm_shared_pingpong(const float* __restrict__ A,
                                     const float* __restrict__ B,
                                     float* __restrict__ C,
                                     int n, int m, int k) {
    __shared__ float As[2][TILE_SIZE][TILE_SIZE + 1];
    __shared__ float Bs[2][TILE_SIZE][TILE_SIZE + 1];

    // Block tiles cover n (rows) and k (cols)
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int n_addr = blockIdx.x * TILE_SIZE + ty; // row in C/A
    const int k_addr = blockIdx.y * TILE_SIZE + tx; // col in C/B

    float acc = 0.f;

    // Prime first tile into buffer 0
    int buf = 0;
    int i = 0;
    if (n_addr < n && (i + tx) < m)
        As[buf][ty][tx] = A[n_addr * m + (i + tx)];
    else
        As[buf][ty][tx] = 0.f;
    if (k_addr < k && (i + ty) < m)
        Bs[buf][ty][tx] = B[(i + ty) * k + k_addr];
    else
        Bs[buf][ty][tx] = 0.f;

    __syncthreads();

    // Main loop with ping-pong buffers (no async copy)
    for (i = 0; i < m; i += TILE_SIZE) {
        // Compute using current buffer
        #pragma unroll
        for (int j = 0; j < TILE_SIZE; ++j) acc += As[buf][ty][j] * Bs[buf][j][tx];

        __syncthreads();

        // Load next tile into the other buffer
        const int next = buf ^ 1;
        const int inext = i + TILE_SIZE;
        if (inext < m) {
            if (n_addr < n && (inext + tx) < m)
                As[next][ty][tx] = A[n_addr * m + (inext + tx)];
            else
                As[next][ty][tx] = 0.f;
            if (k_addr < k && (inext + ty) < m)
                Bs[next][ty][tx] = B[(inext + ty) * k + k_addr];
            else
                Bs[next][ty][tx] = 0.f;
        }

        __syncthreads();
        buf ^= 1;
    }

    if (n_addr < n && k_addr < k)
        C[n_addr * k + k_addr] = acc;
}
