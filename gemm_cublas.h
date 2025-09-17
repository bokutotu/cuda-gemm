#pragma once

#include <cublas_v2.h>
#include "bench_utils.h"

// Row-major GEMM using cuBLAS (column-major API)
// BLAS-style shapes: C[M,N] = A[M,K] x B[K,N]
// Row-major trick: (C_rm)^T = B_rm^T * A_rm^T →
// call col-major: C_col(NxM) = B_col(NxK) * A_col(KxM)
inline void gemm_cublas(cublasHandle_t h,
                        const float* dA, const float* dB, float* dC,
                        int m, int n, int k) {
    const float alpha = 1.0f, beta = 0.0f;
    // In cuBLAS column-major API: C = alpha*op(A)*op(B) + beta*C where
    // A: (N x K) -> use B buffer with no transpose
    // B: (K x M) -> use A buffer with no transpose
    // C: (N x M) -> use C buffer
    checkCublas(
        cublasSgemm(h,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    /*m=*/n, /*n=*/m, /*k=*/k,
                    &alpha,
                    /*A=*/dB, /*lda=*/n,
                    /*B=*/dA, /*ldb=*/k,
                    &beta,
                    /*C=*/dC, /*ldc=*/n),
        "cublasSgemm");
}
