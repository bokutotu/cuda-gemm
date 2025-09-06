#pragma once

inline void matmul(const float* A, const float* B, float* C, int n, int m, int k) {
    for (int i = 0; i < n; ++i) {
        for (int l = 0; l < k; ++l) {
            float sum = 0.0f;
            for (int j = 0; j < m; ++j) {
                sum += A[i*m + j] * B[j*k + l];
            }
            C[i*k + l] = sum;
        }
    }
}
