#include <iostream>
#include <fstream>
#include <cuda_runtime.h>

int main()
{
  std::ofstream log("log.out");

  int deviceCount = 0;
  cudaGetDeviceCount(&deviceCount);

  log << "Number of GPUs : " << deviceCount << "\n\n";

  for(int i = 0; i < deviceCount; i++)
  {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);

    log << "====================================\n";
    log << "GPU ID                : " << i << "\n";
    log << "Name                  : " << prop.name << "\n";
    log << "Compute Capability    : "
        << prop.major << "." << prop.minor << "\n";
    log << "Total Global Memory   : "
        << prop.totalGlobalMem/(1024.0*1024.0*1024.0)
        << " GB\n";
    log << "SM Count              : "
        << prop.multiProcessorCount << "\n";
    log << "Max Threads Per Block : "
        << prop.maxThreadsPerBlock << "\n";
    log << "Warp Size             : "
        << prop.warpSize << "\n";
    log << "Core Clock Rate       : "
        << prop.clockRate/1000.0 << " MHz\n";
    log << "Memory Clock Rate     : "
        << prop.memoryClockRate/1000.0 << " MHz\n";
    log << "Memory Bus Width      : "
        << prop.memoryBusWidth << " bits\n";
    log << "Concurrent Kernels    : "
        << prop.concurrentKernels << "\n";
    log << "Unified Addressing    : "
        << prop.unifiedAddressing << "\n";
    log << "====================================\n\n";
  }

  log.close();
  return 0;
}