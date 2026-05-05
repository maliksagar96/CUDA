#include <iostream>
#include <cuda_runtime.h>
#include <vector>
#include <cassert>
#include <ctime>

using namespace std;

__global__ void add_array(int *a, int *b, int *c, int size) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;

  if (gid < size) {
    c[gid] = a[gid] + b[gid];
  }
}

int main() {

  int size = 1 << 20;
  int block_size = 512;

  size_t byteSize = size * sizeof(int);

  vector<int> h_a(size), h_b(size), gpu_result(size), cpu_result(size);
  
  srand(time(0));  // seed

  for (int i = 0; i < size; i++) {
    h_a[i] = rand() % 1000;   // values between 0–999
    h_b[i] = rand() % 1000;
  }

	clock_t cpu_start, cpu_end;
	cpu_start = clock();

	for(int i = 0;i<size;i++) {
		cpu_result[i] = h_a[i] + h_b[i];
	}

	cpu_end = clock();

  int *d_a, *d_b, *d_c;

  cudaError_t error;

  error = cudaMalloc(&d_a, byteSize);
  if (error != cudaSuccess) {
    cerr << "CUDA error: " << cudaGetErrorString(error) << endl;
    return -1;
  }

  cudaMalloc(&d_b, byteSize);
  cudaMalloc(&d_c, byteSize);

  cudaMemcpy(d_a, h_a.data(), byteSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b.data(), byteSize, cudaMemcpyHostToDevice);

  dim3 block(block_size);
  dim3 grid((size + block_size - 1) / block_size);

  clock_t gpu_start, gpu_end;
  gpu_start = clock();
  add_array<<<grid, block>>>(d_a, d_b, d_c, size);
  gpu_end = clock();

  // check kernel launch error
  error = cudaGetLastError();
  if (error != cudaSuccess) {
    cerr << "Kernel launch error: " << cudaGetErrorString(error) << endl;
    return -1;
  }

  // wait for kernel to finish
  cudaDeviceSynchronize();

  // correct pointer here
  cudaMemcpy(gpu_result.data(), d_c, byteSize, cudaMemcpyDeviceToHost);

  for (int i = 0; i < size; i++) {
    assert(cpu_result[i] == gpu_result[i]);
  }

  cout << "Results match.\n";
  double cputime = (double)(((double)(cpu_end - cpu_start)/CLOCKS_PER_SEC));
  double gputime = (double)(((double)(gpu_end - gpu_start)/CLOCKS_PER_SEC));
  cout << "CPU execution time = "<<cputime<<" seconds."<<endl;
  cout << "GPU execution time = "<<gputime<<" seconds."<<endl;

  cout << "***SPEED UP = "<< cputime/gputime << endl;

  cudaFree(d_a);cudaFree(d_b);cudaFree(d_c);

  return 0;

}