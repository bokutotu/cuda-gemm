#include <cstdio>
#include <cuda_runtime.h>
int main(){
  cudaDeviceProp p{}; int dev=0; cudaGetDevice(&dev); cudaGetDeviceProperties(&p, dev);
  printf("name,%s\n", p.name);
  printf("cc,%d.%d\n", p.major, p.minor);
  printf("sm_count,%d\n", p.multiProcessorCount);
  printf("mem_clock_khz,%d\n", p.memoryClockRate);
  printf("mem_bus_width_bits,%d\n", p.memoryBusWidth);
  printf("global_mem_bytes,%zu\n", (size_t)p.totalGlobalMem);
  printf("regs_per_sm,%d\n", p.regsPerMultiprocessor);
  printf("shared_mem_per_sm,%zu\n", (size_t)p.sharedMemPerMultiprocessor);
  printf("warp_size,%d\n", p.warpSize);
  return 0;
}
