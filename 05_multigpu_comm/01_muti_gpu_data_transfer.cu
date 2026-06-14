#include <iostream>
#include <cuda_runtime.h>

using namespace std;

int main()
{
  int ngpus;

  cudaGetDeviceCount(&ngpus);

  if(ngpus < 2)
  {
    cout << "Need at least 2 GPUs\n";
    return 0;
  }

  int can_access;

  cudaDeviceCanAccessPeer(&can_access, 0, 1);

  if(!can_access)
  {
    cout << "P2P not supported between GPU0 and GPU1\n";
    return 0;
  }

  cudaSetDevice(0);
  cudaDeviceEnablePeerAccess(1, 0);

  cudaSetDevice(1);
  cudaDeviceEnablePeerAccess(0, 0);

  size_t N = 256 * 1024 * 1024; // 256 MB

  char *d_src;
  char *d_dst;

  cudaSetDevice(0);
  cudaMalloc(&d_src, N);

  cudaSetDevice(1);
  cudaMalloc(&d_dst, N);

  cudaEvent_t start, stop;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);

  cudaMemcpyPeer(d_dst,1,d_src,0,N);

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float ms;
  cudaEventElapsedTime(&ms, start, stop);

  double bandwidth = (double)N / (ms * 1.0e-3) / 1.0e9;

  cout << "Transfer size : " << N / (1024.0 * 1024.0) << " MB\n";

  cout << "Time : " << ms << " ms\n";

  cout << "Bandwidth : " << bandwidth << " GB/s\n";

  cudaSetDevice(0);
  cudaFree(d_src);

  cudaSetDevice(1);
  cudaFree(d_dst);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return 0;
}