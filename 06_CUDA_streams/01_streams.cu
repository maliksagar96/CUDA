/*
Que - What will happen if we launch kernel1 and kernel2 using the CUDA streams and without using the CUDA streams?
Ans - Without using the CUDA streams kernel2 will launch only after kernel1 finishes execution. There won't be any resource sharing between kernel1 and kernel2. But if we use the cudastream then kernel 1 and 2 are launched at the same time and they share resources. 

*/


#include <cuda_runtime.h>
#include <iostream>

__global__ void kernel1() {
  printf("Kernel 1\n");
}

__global__ void kernel2() {
  printf("Kernel 2\n");
}

int main() {
  cudaStream_t stream1, stream2;

  cudaStreamCreate(&stream1);
  cudaStreamCreate(&stream2);

  kernel1<<<1, 1, 0, stream1>>>();
  kernel2<<<1, 1, 0, stream2>>>();

  cudaStreamSynchronize(stream1);
  cudaStreamSynchronize(stream2);

  cudaStreamDestroy(stream1);
  cudaStreamDestroy(stream2);

  return 0;
}