#include <iostream>
#include <cuda_runtime.h>

using namespace std;

int main() {

  int deviceCount;
  cudaGetDeviceCount(&deviceCount);

  if (deviceCount == 0) {
    cout << "No CUDA devices found.\n";
    return 0;
  }

  for (int dev = 0; dev < deviceCount; dev++) {

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);

    cout << "Device " << dev << ":\n";

    cout << "Name: " << prop.name << "\n";

    cout << "Compute Capability: "
         << prop.major << "." << prop.minor << "\n";

    cout << "Total Global Memory: "
         << prop.totalGlobalMem / (1024 * 1024) << " MB\n";

    cout << "Max Threads per Block: "
         << prop.maxThreadsPerBlock << "\n";

    cout << "Max Thread Dimensions: "
         << prop.maxThreadsDim[0] << " x "
         << prop.maxThreadsDim[1] << " x "
         << prop.maxThreadsDim[2] << "\n";

    cout << "Max Grid Size: "
         << prop.maxGridSize[0] << " x "
         << prop.maxGridSize[1] << " x "
         << prop.maxGridSize[2] << "\n";

    cout << "Clock Rate: "
         << prop.clockRate / 1000 << " MHz\n";

    cout << "Shared Memory per Block: "
         << prop.sharedMemPerBlock / 1024 << " KB\n";

         cout << "Shared + L1 per SM: "
     << prop.sharedMemPerMultiprocessor / 1024
     << " KB\n";

    cout << "Warp Size: "
         << prop.warpSize << "\n";

    cout << "L2 cache size: "
     << prop.l2CacheSize / (1024.0 * 1024.0)
     << " MB\n";

     cout << "Number of SMs: "
     << prop.multiProcessorCount << "\n";

    cout << "-----------------------------\n";
  }

  return 0;
}