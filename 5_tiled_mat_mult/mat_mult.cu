#include<stdio.h>
#include<stdlib.h>
#include<cuda_runtime.h>
#include<cuda_runtime_api.h>
#include<cuda_device_runtime_api.h>
#include<cuda_runtime_api.h>
#include<cuda_runtime_api.h>

#define TILE_SIZE 16
#define N 1024

__global__ void mat_mult_kernel(float *d_a, float *d_b, float *d_c, int N){
  
}

int main(int argc, char* argv[]){

  float **h_a, **h_b, **h_c;
  float **d_a, **d_b, **d_c;
  
  h_a = (float**)malloc(N * sizeof(float*));
  h_b = (float**)malloc(N * sizeof(float*));
  h_c = (float**)malloc(N * sizeof(float*));
  for(int i = 0; i < N; i++){
    h_a[i] = (float*)malloc(N * sizeof(float));
    h_b[i] = (float*)malloc(N * sizeof(float));
    h_c[i] = (float*)malloc(N * sizeof(float));
  }

  for(int i = 0; i < N; i++){
    for(int j = 0; j < N; j++){
      h_a[i][j] = rand() % 100;
      h_b[i][j] = rand() % 100;
    }
  }

  cudaMalloc((void**)&d_a, N * N * sizeof(float));
  cudaMalloc((void**)&d_b, N * N * sizeof(float));
  cudaMalloc((void**)&d_c, N * N * sizeof(float));

  cudaMemcpy(d_a, h_a, N * N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, N * N * sizeof(float), cudaMemcpyHostToDevice);


  return 0;
}