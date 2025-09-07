#pragma once

#include <vector>
#include <random>
#include <functional>
#include <cstring>
#include <iostream>
#include <iomanip>

#include "bench_utils.h"
#include "gemm_cpu.h"
#include "gemm_naive.cuh"
#include "gemm_shared.cuh"
#include "gemm_shared_float4.cuh"
#include "gemm_shared_pingpong.cuh"
#include "gemm_shared_cp_async.cuh"
#include "gemm_shared_register_tile.cuh"
#include "gemm_cublas.h"

struct GemmDims { int N{1000}, M{1000}, K{1000}; };

struct VariantResult {
    const char* name;
    double time_ms{0};
    size_t mismatches{0};
    float max_abs_err{0};
};

struct GemmBenchData {
    GemmDims d{};
    double cpu_ms{0};              // valid only if CPU reference used
    std::vector<float> ref;        // golden reference (CPU or cuBLAS)
    std::vector<VariantResult> results; // per-kernel results
};

inline GemmBenchData run_benchmarks(const GemmDims d = {}, int seed = 42, bool do_cpu = true) {
    // --- Sizes ---
    const size_t sizeA = static_cast<size_t>(d.N) * d.M;
    const size_t sizeB = static_cast<size_t>(d.M) * d.K;
    const size_t sizeC = static_cast<size_t>(d.N) * d.K;

    // --- Host inputs ---
    std::vector<float> A(sizeA), B(sizeB);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (size_t i = 0; i < sizeA; ++i) A[i] = dist(rng);
    for (size_t i = 0; i < sizeB; ++i) B[i] = dist(rng);

    // --- Output container ---
    GemmBenchData out; out.d = d; out.ref.assign(sizeC, 0.0f);

    // --- Device allocations (single-shot) ---
    float *dA = nullptr, *dB = nullptr, *dC = nullptr;
    checkCuda(cudaMalloc(&dA, sizeA * sizeof(float)), "cudaMalloc dA");
    checkCuda(cudaMalloc(&dB, sizeB * sizeof(float)), "cudaMalloc dB");
    checkCuda(cudaMalloc(&dC, sizeC * sizeof(float)), "cudaMalloc dC");
    checkCuda(cudaMemcpy(dA, A.data(), sizeA * sizeof(float), cudaMemcpyHostToDevice), "A H2D");
    checkCuda(cudaMemcpy(dB, B.data(), sizeB * sizeof(float), cudaMemcpyHostToDevice), "B H2D");

    // --- Common launch geometry ---
    const int TS = TILE_SIZE;
    const dim3 block_naive(16, 16);
    const dim3 grid_naive((d.K + block_naive.x - 1) / block_naive.x,
                          (d.N + block_naive.y - 1) / block_naive.y);
    const dim3 block_shared(TS, TS);
    const dim3 grid_shared((d.N + TS - 1) / TS, (d.K + TS - 1) / TS);
    // Register tile launch geometry (per-thread micro tile REG_TILE_M x REG_TILE_N)
    const dim3 block_reg(TS / REG_TILE_N, TS / REG_TILE_M);
    const dim3 grid_reg = grid_shared;

    // --- cuBLAS handle (reused) ---
    cublasHandle_t handle{};
    checkCublas(cublasCreate(&handle), "cublasCreate");

    // --- Build golden reference ---
    if (do_cpu) {
        out.cpu_ms = benchmark_host_ms([&]{ matmul(A.data(), B.data(), out.ref.data(), d.N, d.M, d.K); });
    } else {
        gemm_cublas(handle, dA, dB, dC, d.N, d.M, d.K);
        checkCuda(cudaMemcpy(out.ref.data(), dC, sizeC * sizeof(float), cudaMemcpyDeviceToHost), "golden D2H");
    }

    // --- Task abstraction: name + launch() ---
    struct Task {
        const char* name;
        std::function<void()> launch;
    };

    std::vector<Task> tasks;
    tasks.reserve(11);

    tasks.push_back(Task{ "gemm_naive", [=]() {
        gemm_naive<<<grid_naive, block_naive>>>(dA, dB, dC, d.N, d.M, d.K);
    }});
    tasks.push_back(Task{ "gemm_shared", [=]() {
        gemm_shared<<<grid_shared, block_shared>>>(dA, dB, dC, d.N, d.M, d.K);
    }});
    tasks.push_back(Task{ "gemm_shared_float4", [=]() {
        gemm_shared_float4<<<grid_shared, block_shared>>>(dA, dB, dC, d.N, d.M, d.K);
    }});
    tasks.push_back(Task{ "gemm_shared_pingpong", [=]() {
        gemm_shared_pingpong<<<grid_shared, block_shared>>>(dA, dB, dC, d.N, d.M, d.K);
    }});
    tasks.push_back(Task{ "gemm_shared_pingpong_float4", [=]() {
        gemm_shared_pingpong_float4<<<grid_shared, block_shared>>>(dA, dB, dC, d.N, d.M, d.K);
    }});
    tasks.push_back(Task{ "gemm_shared_cp_async", [=]() {
        gemm_shared_cp_async<<<grid_shared, block_shared>>>(dA, dB, dC, d.N, d.M, d.K);
    }});
    tasks.push_back(Task{ "gemm_shared_cp_async_register_tile", [=]() {
        gemm_shared_cp_async_register_tile<<<grid_reg, block_reg>>>(dA, dB, dC, d.N, d.M, d.K);
    }});
    tasks.push_back(Task{ "cublasSgemm (gemm_cublas)", [=]() {
        gemm_cublas(handle, dA, dB, dC, d.N, d.M, d.K);
    }});
    // Register-tile variants
    tasks.push_back(Task{ "gemm_shared_register_tile", [=]() {
        gemm_shared_register_tile<<<grid_reg, block_reg>>>(dA, dB, dC, d.N, d.M, d.K);
    }});
    tasks.push_back(Task{ "gemm_shared_register_tile_float4", [=]() {
        gemm_shared_register_tile_float4<<<grid_reg, block_reg>>>(dA, dB, dC, d.N, d.M, d.K);
    }});
    tasks.push_back(Task{ "gemm_shared_register_tile_pingpong", [=]() {
        gemm_shared_register_tile_pingpong<<<grid_reg, block_reg>>>(dA, dB, dC, d.N, d.M, d.K);
    }});
    tasks.push_back(Task{ "gemm_shared_register_tile_pingpong_float4", [=]() {
        gemm_shared_register_tile_pingpong_float4<<<grid_reg, block_reg>>>(dA, dB, dC, d.N, d.M, d.K);
    }});

    // --- Execute, time, and validate each variant against golden ---
    std::vector<float> hostC(sizeC);
    out.results.reserve(tasks.size());
    for (auto& t : tasks) {
        VariantResult r{}; r.name = t.name;
        r.time_ms = benchmark_gpu_ms([&]{ t.launch(); });
        checkCuda(cudaMemcpy(hostC.data(), dC, sizeC * sizeof(float), cudaMemcpyDeviceToHost), t.name);
        r.mismatches = validate_atol(out.ref, hostC, 1e-2f, r.max_abs_err);
        out.results.push_back(r);
    }

    checkCublas(cublasDestroy(handle), "cublasDestroy");
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    return out;
}

// Pretty-print results in a structured, aligned table.
inline void print_results(const GemmBenchData& data, std::ostream& os = std::cout) {
    const auto& d = data.d;
    os << "=== GEMM Benchmark ===\n";
    os << "Dims: N=" << d.N << " M=" << d.M << " K=" << d.K << "\n";
    if (data.cpu_ms > 0.0) {
        os << "Reference: CPU (avg)     : " << std::fixed << std::setprecision(3) << data.cpu_ms << " ms\n";
    } else {
        os << "Reference: cuBLAS (avg)  : used (CPU skipped)\n";
    }
    // Column widths
    size_t name_w = 6; // min for header
    for (const auto& r : data.results) name_w = std::max(name_w, std::strlen(r.name));
    const int w_name = static_cast<int>(name_w);
    const int w_time = 12;
    const int w_mis  = 12;
    const int w_err  = 12;
    const int w_gflops = 12;

    os << "Results:" << '\n';
    os << "  " << std::left << std::setw(w_name) << "Kernel"
       << "  " << std::right << std::setw(w_time) << "Time (ms)"
       << "  " << std::right << std::setw(w_gflops) << "GFLOPS"
       << "  " << std::right << std::setw(w_mis)  << "Mismatches"
       << "  " << std::right << std::setw(w_err)  << "Max|Err|" << '\n';

    const int total_w = 2 + w_name + 2 + w_time + 2 + w_gflops + 2 + w_mis + 2 + w_err;
    os << "  " << std::string(total_w - 2, '-') << '\n';

    const double ops = 2.0 * static_cast<double>(d.N) * d.M * d.K; // FMA counted as 2 ops
    for (const auto& r : data.results) {
        const double gflops = (r.time_ms > 0.0) ? (ops / 1.0e9) / (r.time_ms / 1.0e3) : 0.0;
        os << "  " << std::left  << std::setw(w_name) << r.name
           << "  " << std::right << std::setw(w_time) << std::fixed << std::setprecision(3) << r.time_ms
           << "  " << std::right << std::setw(w_gflops) << std::fixed << std::setprecision(1) << gflops
           << "  " << std::right << std::setw(w_mis)  << r.mismatches
           << "  " << std::right << std::setw(w_err)  << std::scientific << std::setprecision(2) << r.max_abs_err
           << std::defaultfloat << '\n';
    }
}
