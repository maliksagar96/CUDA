#include <stdio.h>
#include <cuda_runtime.h>

__global__ void naive_prefix_sum(int* d_a, int* d_prefix_sum, int n){

  int i = threadIdx.x;
  if(i<n) {
    int sum = 0;
    for(int j = 0;j<=i;j++) {
      sum += d_a[j];
    }
    d_prefix_sum[i] = sum;
  }
}

int main() {
  int n = 10;
  int* a = (int*)malloc(sizeof(int) * n);
  int* prefix_sum = (int*)malloc(sizeof(int) * n);

  int *d_a,*d_prefix_sum;

  cudaMalloc((void**)&d_a, sizeof(int)*n);
  cudaMalloc((void**)&d_prefix_sum, sizeof(int)*n);

  for (int i = 0; i < n; i++) {
    a[i] = i;
  }

  cudaMemcpy(d_a, a, sizeof(int)*n, cudaMemcpyHostToDevice);

  naive_prefix_sum<<<1,n>>>(d_a, d_prefix_sum, n);
  cudaMemcpy(prefix_sum, d_prefix_sum, sizeof(int)*n, cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  
  for(int i = 0;i<n;i++) {
    printf("prefix sum[%d] = %d\n", i, prefix_sum[i]);
  }

  free(a);
  free(prefix_sum);
  cudaFree(d_a);
  cudaFree(d_prefix_sum);
}