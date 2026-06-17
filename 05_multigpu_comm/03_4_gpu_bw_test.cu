/*
  First check if there are 4 GPUs there or not.
  Then it checks the bandwidht of all three pairs. 
  Then the protocol asks if the src gpu can access the memory of the destination GPU.
  Then we have to set the GPU on which we need to allocate memory. 
  Then we do the cudamemcpy from source to device. 

*/

#include <iostream>
#include <iomanip>
#include <cuda_runtime.h>

using namespace std;

double measure_bandwidth(int src_gpu, int dst_gpu, size_t bytes) {

  int can_access;
  cudaDeviceCanAccessPeer(&can_access, src_gpu, dst_gpu);

  if(!can_access)
    return -1.0;

  cudaSetDevice(src_gpu);
  cudaDeviceEnablePeerAccess(dst_gpu, 0);

  cudaSetDevice(dst_gpu);
  cudaDeviceEnablePeerAccess(src_gpu, 0);

  char *d_src;
  char *d_dst;

  cudaSetDevice(src_gpu);
  cudaMalloc(&d_src, bytes);

  cudaSetDevice(dst_gpu);
  cudaMalloc(&d_dst, bytes);

  cudaEvent_t start, stop;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);

  cudaMemcpyPeer( d_dst, dst_gpu, d_src, src_gpu, bytes);

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float ms;

  cudaEventElapsedTime(&ms, start, stop);

  double bandwidth = (double)bytes / (ms * 1.0e-3) / 1.0e9;

  cudaSetDevice(src_gpu);
  cudaFree(d_src);

  cudaSetDevice(dst_gpu);
  cudaFree(d_dst);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return bandwidth;
}

int main() {
  
  int ngpus;
  cudaGetDeviceCount(&ngpus);

  if(ngpus < 4) {
    cout << "Need at least 4 GPUs\n";
    return 0;
  }

  size_t bytes = 256ULL * 1024 * 1024;
  cout << fixed << setprecision(2);

  for(int src = 0; src < 4; src++) {
    for(int dst = 0; dst < 4; dst++) {
      if(src == dst)
        continue;

      double bw = measure_bandwidth(src, dst, bytes);
      if(bw < 0.0)  cout << "GPU" << src << " -> GPU" << dst << " : P2P not supported\n";
      else cout << "GPU"<< src<< " -> GPU"<< dst<< " : "<< bw<< " GB/s\n";
    }
  }

  return 0;
}