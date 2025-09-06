#pragma once

#include <vector>
#include <random>

#include "bench_utils.h"
#include "gemm_cpu.h"
#include "gemm_naive.cuh"
#include "gemm_shared.cuh"
#include "gemm_shared_float4.cuh"
#include "gemm_cublas.h"

struct GemmDims { int N{1000}, M{1000}, K{1000}; };

struct GemmBenchData {
    GemmDims d{};
    std::vector<float> C_cpu;
    std::vector<float> C_naive;
    std::vector<float> C_shared;
    std::vector<float> C_shared4;
    std::vector<float> C_cublas;
    double cpu_ms{0}, naive_ms{0}, shared_ms{0}, shared4_ms{0}, cublas_ms{0};
};

inline GemmBenchData run_benchmarks(const GemmDims d = {}, int seed = 42, bool do_cpu = true) {
    const size_t sizeA = static_cast<size_t>(d.N) * d.M;
    const size_t sizeB = static_cast<size_t>(d.M) * d.K;
    const size_t sizeC = static_cast<size_t>(d.N) * d.K;

    // Host data
    std::vector<float> A(sizeA), B(sizeB);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (size_t i = 0; i < sizeA; ++i) A[i] = dist(rng);
    for (size_t i = 0; i < sizeB; ++i) B[i] = dist(rng);

    GemmBenchData out; out.d = d;
    out.C_cpu.assign(sizeC, 0.0f);
    out.C_naive.assign(sizeC, 0.0f);
    out.C_shared.assign(sizeC, 0.0f);
    out.C_shared4.assign(sizeC, 0.0f);
    out.C_cublas.assign(sizeC, 0.0f);

    // CPU reference (optional)
    if (do_cpu) {
        out.cpu_ms = benchmark_host_ms([&]{ matmul(A.data(), B.data(), out.C_cpu.data(), d.N, d.M, d.K); });
    }

    // Device allocations
    float *dA = nullptr, *dB = nullptr, *dC = nullptr;
    checkCuda(cudaMalloc(&dA, sizeA * sizeof(float)), "cudaMalloc dA");
    checkCuda(cudaMalloc(&dB, sizeB * sizeof(float)), "cudaMalloc dB");
    checkCuda(cudaMalloc(&dC, sizeC * sizeof(float)), "cudaMalloc dC");
    checkCuda(cudaMemcpy(dA, A.data(), sizeA * sizeof(float), cudaMemcpyHostToDevice), "A H2D");
    checkCuda(cudaMemcpy(dB, B.data(), sizeB * sizeof(float), cudaMemcpyHostToDevice), "B H2D");

    // Naive kernel
    dim3 block_naive(16, 16);
    dim3 grid_naive((d.K + block_naive.x - 1) / block_naive.x,
                    (d.N + block_naive.y - 1) / block_naive.y);
    checkCuda(cudaMemset(dC, 0, sizeC * sizeof(float)), "memset C naive");
    out.naive_ms = benchmark_gpu_ms([&]{ gemm_naive<<<grid_naive, block_naive>>>(dA, dB, dC, d.N, d.M, d.K); });
    checkCuda(cudaMemcpy(out.C_naive.data(), dC, sizeC * sizeof(float), cudaMemcpyDeviceToHost), "C_naive D2H");

    // Shared kernel
    const int TS = TILE_SIZE;
    dim3 block_shared(TS, TS);
    dim3 grid_shared((d.N + TS - 1) / TS, (d.K + TS - 1) / TS);
    checkCuda(cudaMemset(dC, 0, sizeC * sizeof(float)), "memset C shared");
    out.shared_ms = benchmark_gpu_ms([&]{ gemm_shared<<<grid_shared, block_shared>>>(dA, dB, dC, d.N, d.M, d.K); });
    checkCuda(cudaMemcpy(out.C_shared.data(), dC, sizeC * sizeof(float), cudaMemcpyDeviceToHost), "C_shared D2H");

    // Shared float4 kernel
    checkCuda(cudaMemset(dC, 0, sizeC * sizeof(float)), "memset C shared4");
    out.shared4_ms = benchmark_gpu_ms([&]{ gemm_shared_float4<<<grid_shared, block_shared>>>(dA, dB, dC, d.N, d.M, d.K); });
    checkCuda(cudaMemcpy(out.C_shared4.data(), dC, sizeC * sizeof(float), cudaMemcpyDeviceToHost), "C_shared4 D2H");

    // cuBLAS SGEMM (row-major mapping)
    cublasHandle_t handle{};
    checkCublas(cublasCreate(&handle), "cublasCreate");
    // Use default stream (0); benchmark_gpu_ms records events on stream 0
    checkCuda(cudaMemset(dC, 0, sizeC * sizeof(float)), "memset C cublas");
    out.cublas_ms = benchmark_gpu_ms([&]{
        gemm_cublas(handle, dA, dB, dC, d.N, d.M, d.K);
    });
    checkCuda(cudaMemcpy(out.C_cublas.data(), dC, sizeC * sizeof(float), cudaMemcpyDeviceToHost), "C_cublas D2H");
    checkCublas(cublasDestroy(handle), "cublasDestroy");

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    return out;
}

struct ValidationSummary {
    size_t mis_cpu_naive{0}, mis_cpu_shared{0}, mis_naive_shared{0};
    size_t mis_cpu_shared4{0}, mis_naive_shared4{0}, mis_shared_shared4{0};
    size_t mis_cpu_cublas{0}, mis_naive_cublas{0}, mis_shared_cublas{0}, mis_shared4_cublas{0};
    float err_cpu_naive{0}, err_cpu_shared{0}, err_naive_shared{0};
    float err_cpu_shared4{0}, err_naive_shared4{0}, err_shared_shared4{0};
    float err_cpu_cublas{0}, err_naive_cublas{0}, err_shared_cublas{0}, err_shared4_cublas{0};
    bool ok{false};
};

inline ValidationSummary validate_outputs(const GemmBenchData& out, float atol = 1e-2f, bool have_cpu = true) {
    ValidationSummary s{};
    if (have_cpu) {
        s.mis_cpu_naive = validate_atol(out.C_cpu, out.C_naive, atol, s.err_cpu_naive);
        s.mis_cpu_shared = validate_atol(out.C_cpu, out.C_shared, atol, s.err_cpu_shared);
        s.mis_cpu_shared4 = validate_atol(out.C_cpu, out.C_shared4, atol, s.err_cpu_shared4);
        s.mis_cpu_cublas = validate_atol(out.C_cpu, out.C_cublas, atol, s.err_cpu_cublas);
    } else {
        s.mis_cpu_naive = s.mis_cpu_shared = s.mis_cpu_shared4 = s.mis_cpu_cublas = 0;
        s.err_cpu_naive = s.err_cpu_shared = s.err_cpu_shared4 = s.err_cpu_cublas = 0.0f;
    }
    s.mis_naive_shared = validate_atol(out.C_naive, out.C_shared, atol, s.err_naive_shared);
    s.mis_naive_shared4 = validate_atol(out.C_naive, out.C_shared4, atol, s.err_naive_shared4);
    s.mis_shared_shared4 = validate_atol(out.C_shared, out.C_shared4, atol, s.err_shared_shared4);
    s.mis_naive_cublas   = validate_atol(out.C_naive,   out.C_cublas,  atol, s.err_naive_cublas);
    s.mis_shared_cublas  = validate_atol(out.C_shared,  out.C_cublas,  atol, s.err_shared_cublas);
    s.mis_shared4_cublas = validate_atol(out.C_shared4, out.C_cublas,  atol, s.err_shared4_cublas);
    s.ok = (s.mis_naive_shared == 0 && s.mis_naive_shared4 == 0 && s.mis_shared_shared4 == 0 &&
            s.mis_naive_cublas == 0 && s.mis_shared_cublas == 0 && s.mis_shared4_cublas == 0)
           && (!have_cpu || (s.mis_cpu_naive == 0 && s.mis_cpu_shared == 0 && s.mis_cpu_shared4 == 0 && s.mis_cpu_cublas == 0));
    return s;
}
