#pragma once

#include <cublas_v2.h>
#include "bench_utils.h"

// Row-major GEMM using cuBLAS (which is column-major):
// C[n,k] = A[n,m] x B[m,k]
// We compute this by treating row-major buffers as column-major of transposed shapes.
// Mapping: call C_col(KxN) = B_col(KxM) * A_col(MxN)
inline void gemm_cublas(cublasHandle_t h,
                        const float* dA, const float* dB, float* dC,
                        int n, int m, int k) {
    const float alpha = 1.0f, beta = 0.0f;
    // In cuBLAS column-major API: C = alpha*op(A)*op(B) + beta*C where
    // A: (K x M) -> use B buffer with no transpose
    // B: (M x N) -> use A buffer with no transpose
    // C: (K x N) -> use C buffer
    checkCublas(
        cublasSgemm(h,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    /*m=*/k, /*n=*/n, /*k=*/m,
                    &alpha,
                    /*A=*/dB, /*lda=*/k,
                    /*B=*/dA, /*ldb=*/m,
                    &beta,
                    /*C=*/dC, /*ldc=*/k),
        "cublasSgemm");
}

