/*
Since this is my first CUDA code I wanted to check the number of threads and the maximum number of gridsize. 
Use the following to check:

cudaDeviceProp prop;
cudaGetDeviceProperties(&prop, 0);

printf("Max Threads per block = %d\n", prop.maxThreadsPerBlock);


printf("Max grid size: %d x %d x %d\n",
      prop.maxGridSize[0],  // max blocks in x-dimension
      prop.maxGridSize[1],  // max blocks in y-dimension
      prop.maxGridSize[2]); // max blocks in z-dimension

Note:We can keep number of threads 256 and the blocksPerGrid = (N + threadsPerBlock -1)/threadsPerBlock. 
We can change the number of threads from 256 to 128, 512 or 1024 etc. if we are looking for extreme optimisation. 

*/

#include <stdio.h> 
#include <cuda_runtime.h>
#include <stdlib.h>

//Vector size
#define N 1000000

//Vector addition using CPU.
void vector_add_cpu(float *a, float *b, float* c) {
  for(int i = 0;i<N;i++) {
    c[i] = a[i] + b[i];
  }
}

//Vector addition using GPU.
__global__ void vector_add_gpu(float* a, float* b, float* c, int n) {
 int threadID = blockIdx.x * blockDim.x + threadIdx.x;
 if(threadID < N) {
  c[threadID] = a[threadID] + b[threadID];
 }
}

int main() {

  //Declaring host and device variables
  float *h_a, *h_b, *h_c;
  float *d_a, *d_b, *d_c;
  size_t size = N * sizeof(float);

  //Allocate host memory
  h_a = (float*)malloc(size);
  h_b = (float*)malloc(size);
  h_c = (float*)malloc(size);

  // Allocate device memory
  cudaMalloc(&d_a, size);
  cudaMalloc(&d_b, size);
  cudaMalloc(&d_c, size);

  //Using a constant random seed. 
  srand(42);
  for (int i = 0; i < N; i++) {
      h_a[i] = rand() / (float)RAND_MAX;
      h_b[i] = rand() / (float)RAND_MAX;
  }

  
  // vector_add_cpu(h_a,h_b,h_c);

  // for (int i = 0; i < 5; i++) {
  //   h_c[i] = h_a[i] + h_b[i];
  //   printf("%f + %f = %f\n", h_a[i], h_b[i], h_c[i]);
  // }

  cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);
 
  int threadsPerBlock = 256;
  int blocksPerGrid = (N + threadsPerBlock -1)/threadsPerBlock;

  vector_add_gpu<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);
  cudaDeviceSynchronize();
  cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

  // for (int i = 0; i < 5; i++) {
  //   // h_c[i] = h_a[i] + h_b[i];
  //   printf("%f + %f = %f\n", h_a[i], h_b[i], h_c[i]);
  // }

  free(h_a);
  free(h_b);
  free(h_c);
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  return 0;
}