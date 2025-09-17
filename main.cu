// Keep main simple: call benchmark and validation functions only.
#include <iostream>
#include <vector>
#include <cstring>
#include "gemm_bench.h"

static void usage(const char* prog){
    std::cout << "Usage: " << prog << " [M [N K]] [--no-cpu]\n"
              << "  - if only M given, uses M=N=K=M\n"
              << "  - default M=N=K=1000\n"
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
        int s = std::atoi(args[0]);
        if (s > 0) dims = GemmDims{s, s, s};
    } else if (args.size() == 3) {
        int M = std::atoi(args[0]);
        int N = std::atoi(args[1]);
        int K = std::atoi(args[2]);
        if (M>0 && N>0 && K>0) dims = GemmDims{M, N, K};
    } else if (!args.empty()) {
        usage(argv[0]);
        return 1;
    }

    const auto data = run_benchmarks(dims, /*seed=*/42, /*do_cpu=*/!no_cpu);

    print_results(data);

    bool ok = true;
    for (const auto& r : data.results) ok = ok && (r.mismatches == 0);
    return ok ? 0 : 1;
}
