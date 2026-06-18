#include <iostream>
#include <fstream>
#include <cuda_runtime.h>

int main()
{
  std::ofstream log("log.out");
  int deviceCount=0;
  cudaGetDeviceCount(&deviceCount);

  std::cout<<"Number of GPUs : "<<deviceCount<<"\n\n";
  log<<"Number of GPUs : "<<deviceCount<<"\n\n";

  for(int i=0;i<deviceCount;i++)
  {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop,i);

    std::cout<<"====================================\n";
    std::cout<<"GPU ID                : "<<i<<"\n";
    std::cout<<"Name                  : "<<prop.name<<"\n";
    std::cout<<"Compute Capability    : "<<prop.major<<"."<<prop.minor<<"\n";
    std::cout<<"Total Global Memory   : "<<prop.totalGlobalMem/(1024.0*1024.0*1024.0)<<" GB\n";
    std::cout<<"SM Count              : "<<prop.multiProcessorCount<<"\n";
    std::cout<<"Max Threads Per Block : "<<prop.maxThreadsPerBlock<<"\n";
    std::cout<<"Warp Size             : "<<prop.warpSize<<"\n";
    std::cout<<"Memory Bus Width      : "<<prop.memoryBusWidth<<" bits\n";
    std::cout<<"Concurrent Kernels    : "<<prop.concurrentKernels<<"\n";
    std::cout<<"Unified Addressing    : "<<prop.unifiedAddressing<<"\n";
    std::cout<<"====================================\n\n";

    log<<"====================================\n";
    log<<"GPU ID                : "<<i<<"\n";
    log<<"Name                  : "<<prop.name<<"\n";
    log<<"Compute Capability    : "<<prop.major<<"."<<prop.minor<<"\n";
    log<<"Total Global Memory   : "<<prop.totalGlobalMem/(1024.0*1024.0*1024.0)<<" GB\n";
    log<<"SM Count              : "<<prop.multiProcessorCount<<"\n";
    log<<"Max Threads Per Block : "<<prop.maxThreadsPerBlock<<"\n";
    log<<"Warp Size             : "<<prop.warpSize<<"\n";
    log<<"Memory Bus Width      : "<<prop.memoryBusWidth<<" bits\n";
    log<<"Concurrent Kernels    : "<<prop.concurrentKernels<<"\n";
    log<<"Unified Addressing    : "<<prop.unifiedAddressing<<"\n";
    log<<"====================================\n\n";
  }

  log.close();
  return 0;
}