// Keep main simple: call benchmark and validation functions only.
#include <iostream>
#include <vector>
#include <cstring>
#include "gemm_bench.h"

static void usage(const char* prog){
    std::cout << "Usage: " << prog << " [N [M K]] [--no-cpu]\n"
              << "  - if only N given, uses N=N, M=N, K=N\n"
              << "  - default N=M=K=1000\n"
              << "  - --no-cpu : skip CPU reference (recommended for large sizes)\n";
}

int main(int argc, char** argv) {
    GemmDims dims{1000, 1000, 1000};
    bool no_cpu = false;

    // Parse args: optional N [M K] and flag --no-cpu
    std::vector<const char*> args;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
            usage(argv[0]);
            return 0;
        } else if (std::strcmp(argv[i], "--no-cpu") == 0) {
            no_cpu = true;
        } else {
            args.push_back(argv[i]);
        }
    }
    if (args.size() == 1) {
        int n = std::atoi(args[0]);
        if (n > 0) dims = GemmDims{n, n, n};
    } else if (args.size() == 3) {
        int N = std::atoi(args[0]);
        int M = std::atoi(args[1]);
        int K = std::atoi(args[2]);
        if (N>0 && M>0 && K>0) dims = GemmDims{N, M, K};
    } else if (!args.empty()) {
        usage(argv[0]);
        return 1;
    }

    std::cout << "Dims: N=" << dims.N << " M=" << dims.M << " K=" << dims.K
              << (no_cpu ? " (CPU skipped)" : "") << "\n";

    const auto data = run_benchmarks(dims, /*seed=*/42, /*do_cpu=*/!no_cpu);
    const auto val  = validate_outputs(data, 1e-2f, /*have_cpu=*/!no_cpu);

    if (!no_cpu) {
        std::cout << "CPU time (avg): " << data.cpu_ms << " ms\n";
    } else {
        std::cout << "CPU time (avg): (skipped)\n";
    }
    std::cout << "GPU naive time (avg): " << data.naive_ms << " ms\n";
    std::cout << "GPU shared time (avg): " << data.shared_ms << " ms\n";
    std::cout << "GPU shared(float4) time (avg): " << data.shared4_ms << " ms\n";
    std::cout << "GPU cuBLAS time (avg): " << data.cublas_ms << " ms\n";

    if (!no_cpu) {
        std::cout << "CPU vs Naive -> mismatches=" << val.mis_cpu_naive
                  << ", max_abs_err=" << val.err_cpu_naive << "\n";
        std::cout << "CPU vs Shared -> mismatches=" << val.mis_cpu_shared
                  << ", max_abs_err=" << val.err_cpu_shared << "\n";
        std::cout << "CPU vs Shared4 -> mismatches=" << val.mis_cpu_shared4
                  << ", max_abs_err=" << val.err_cpu_shared4 << "\n";
        std::cout << "CPU vs cuBLAS  -> mismatches=" << val.mis_cpu_cublas
                  << ", max_abs_err=" << val.err_cpu_cublas << "\n";
    }
    std::cout << "Naive vs Shared -> mismatches=" << val.mis_naive_shared
              << ", max_abs_err=" << val.err_naive_shared << std::endl;
    std::cout << "Naive vs Shared4 -> mismatches=" << val.mis_naive_shared4
              << ", max_abs_err=" << val.err_naive_shared4 << "\n";
    std::cout << "Shared vs Shared4 -> mismatches=" << val.mis_shared_shared4
              << ", max_abs_err=" << val.err_shared_shared4 << std::endl;
    std::cout << "Naive vs cuBLAS  -> mismatches=" << val.mis_naive_cublas
              << ", max_abs_err=" << val.err_naive_cublas << "\n";
    std::cout << "Shared vs cuBLAS -> mismatches=" << val.mis_shared_cublas
              << ", max_abs_err=" << val.err_shared_cublas << "\n";
    std::cout << "Shared4 vs cuBLAS-> mismatches=" << val.mis_shared4_cublas
              << ", max_abs_err=" << val.err_shared4_cublas << std::endl;

    return val.ok ? 0 : 1;
}
