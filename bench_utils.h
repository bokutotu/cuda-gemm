#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <algorithm>
#include <vector>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>

// Lightweight CUDA error check
static inline void checkCuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA error at %s: %s\n", msg, cudaGetErrorString(err));
        std::abort();
    }
}

// Lightweight cuBLAS error check
static inline const char* cublasStatusString(cublasStatus_t s) {
    switch (s) {
        case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
        case CUBLAS_STATUS_NOT_SUPPORTED: return "CUBLAS_STATUS_NOT_SUPPORTED";
        case CUBLAS_STATUS_LICENSE_ERROR: return "CUBLAS_STATUS_LICENSE_ERROR";
        default: return "CUBLAS_STATUS_UNKNOWN";
    }
}

static inline void checkCublas(cublasStatus_t stat, const char* msg) {
    if (stat != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "cuBLAS error at %s: %s (%d)\n", msg, cublasStatusString(stat), (int)stat);
        std::abort();
    }
}

// Benchmark a host-side callable (CPU), returns average ms over iters (after warmup)
template <typename F>
inline double benchmark_host_ms(F&& fn, int warmup = 0, int iters = 1) {
    for (int i = 0; i < warmup; ++i) fn();
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) fn();
    auto t1 = std::chrono::high_resolution_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return total_ms / std::max(1, iters);
}

// Benchmark a GPU kernel launch callable using CUDA events; returns average ms
template <typename F>
inline double benchmark_gpu_ms(F&& launch, int warmup = 5, int iters = 10) {
    // warmup
    launch();
    checkCuda(cudaGetLastError(), "kernel launch (warmup)");
    checkCuda(cudaDeviceSynchronize(), "kernel sync (warmup)");
    for (int i = 1; i < warmup; ++i) launch();
    checkCuda(cudaDeviceSynchronize(), "kernel sync (post-warmup)");

    cudaEvent_t start, stop;
    checkCuda(cudaEventCreate(&start), "event create start");
    checkCuda(cudaEventCreate(&stop), "event create stop");
    float total_ms = 0.0f;
    for (int i = 0; i < iters; ++i) {
        checkCuda(cudaEventRecord(start), "event record start");
        launch();
        checkCuda(cudaEventRecord(stop), "event record stop");
        checkCuda(cudaEventSynchronize(stop), "event sync stop");
        float ms = 0.0f;
        checkCuda(cudaEventElapsedTime(&ms, start, stop), "event elapsed time");
        total_ms += ms;
    }
    checkCuda(cudaEventDestroy(start), "event destroy start");
    checkCuda(cudaEventDestroy(stop), "event destroy stop");
    return static_cast<double>(total_ms) / std::max(1, iters);
}

// Validate two vectors with absolute tolerance; returns mismatch count and sets max_abs_err
template <typename T>
inline size_t validate_atol(const std::vector<T>& ref, const std::vector<T>& got, T atol, T& max_abs_err) {
    assert(ref.size() == got.size());
    size_t mismatches = 0;
    max_abs_err = static_cast<T>(0);
    for (size_t i = 0; i < ref.size(); ++i) {
        T diff = std::fabs(ref[i] - got[i]);
        if (diff > max_abs_err) max_abs_err = diff;
        if (diff > atol) ++mismatches;
    }
    return mismatches;
}
