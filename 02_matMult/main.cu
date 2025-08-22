 #include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define N 10
#define M 10

void matmult_cpu(float* a, float*b, float* c){

  for(int i= 0;i<N;i++) {
    for(int j = 0;j<M;j++){
      c[i*M + j] = 0;
      for(int k = 0;k<N;k++) {
        c[i*M + j] += a[i*N + k] * b[k*M + j];
      }      
    }
  }
}

__global__ void matmult(float* a, float* b, float* c, int n, int m) {
  
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = n*m;

  if(idx < total) {
    int i = idx/m;
    int j = idx%m;
    float sum = 0;

    for(int k = 0;k<n;k++) {
      //sum += a[i][k] * b[k][j];
      sum += a[i*n + k] * b[k*m + j];
    }
    //c[i][j] = sum;
    c[i * m + j] = sum;
  }

  
}

int main() {

  //Declaring single column or flattened matrices.
  //The i,j th element of a NxM flatted matrix can be obtained using a[i*M + j].
  float* h_a = (float*)malloc(N*M*sizeof(float));
  float* h_b = (float*)malloc(N*M*sizeof(float));
  float* h_c = (float*)malloc(N*M*sizeof(float));
  
  srand(42);
  for(int i = 0;i<M*N;i++) {
    h_a[i] = rand()/(float)RAND_MAX;
    h_b[i] = rand()/(float)RAND_MAX;
  }

  float *d_a, *d_b, *d_c;

  cudaMalloc((void**)&d_a, N*M*sizeof(float));
  cudaMalloc((void**)&d_b, N*M*sizeof(float));
  cudaMalloc((void**)&d_c, N*M*sizeof(float));

  cudaMemcpy(d_a, h_a, M*N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, M*N*sizeof(float), cudaMemcpyHostToDevice);

  int threadsPerBlock = 256;
  int blocksPerGrid = (N*M + threadsPerBlock - 1)/threadsPerBlock;

  matmult<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N, M);

  cudaMemcpy(h_c, d_c, M*N*sizeof(float), cudaMemcpyDeviceToHost);

  free(h_a);
  free(h_b);
  free(h_c);
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  return 0;
}