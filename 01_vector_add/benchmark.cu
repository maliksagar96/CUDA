#include <stdio.h> 
#include <cuda_runtime.h>
#include <stdlib.h>
#include <chrono>
#include <cmath> // for fabs
using namespace std::chrono;

#define N 1000000

void vector_add_cpu(float *a, float *b, float* c) {
  for (int i = 0; i < N; i++) {
    c[i] = a[i] + b[i];
  }
}

__global__ void vector_add_gpu(float* a, float* b, float* c, int n) {
  int threadID = blockIdx.x * blockDim.x + threadIdx.x;
  if (threadID < n) {
    c[threadID] = a[threadID] + b[threadID];
  }
}

int main() {
  float *h_a, *h_b, *h_c, *h_c_gpu;
  float *d_a, *d_b, *d_c;
  size_t size = N * sizeof(float);

  double cpu_duration = 0;
  double gpu_duration = 0;

  h_a = (float*)malloc(size);
  h_b = (float*)malloc(size);
  h_c = (float*)malloc(size);
  h_c_gpu = (float*)malloc(size);  // extra array to store GPU results

  cudaMalloc(&d_a, size);
  cudaMalloc(&d_b, size);
  cudaMalloc(&d_c, size);

  int threadsPerBlock = 256;
  int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

  for (int itr = 0; itr < 20; itr++) {
    srand(itr);
    for (int i = 0; i < N; i++) {
      h_a[i] = rand() / (float)RAND_MAX;
      h_b[i] = rand() / (float)RAND_MAX;
    }

    auto start = high_resolution_clock::now();
    vector_add_cpu(h_a, h_b, h_c);
    auto end = high_resolution_clock::now();
    cpu_duration += duration_cast<microseconds>(end - start).count();
    // printf("CPU duration: %.0f microseconds\n", cpu_duration);
  }

  for (int itr = 0; itr < 20; itr++) {
    srand(itr);
    for (int i = 0; i < N; i++) {
      h_a[i] = rand() / (float)RAND_MAX;
      h_b[i] = rand() / (float)RAND_MAX;
    }

    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    auto start = high_resolution_clock::now();
    vector_add_gpu<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);
    cudaDeviceSynchronize();  // wait for GPU to finish
    auto end = high_resolution_clock::now();
    gpu_duration += duration_cast<microseconds>(end - start).count();
    // printf("GPU duration: %.0f microseconds\n", gpu_duration);

    cudaMemcpy(h_c_gpu, d_c, size, cudaMemcpyDeviceToHost);

    // Compare CPU and GPU results
    bool match = true;
    double epsilon = 1e-5;
    for (int i = 0; i < N; i++) {
      if (fabs(h_c[i] - h_c_gpu[i]) > epsilon) {
        printf("Mismatch at index %d: CPU = %f, GPU = %f\n", i, h_c[i], h_c_gpu[i]);
        match = false;        
        break;
      }
    }
    if (match)
      printf("Results match for iteration %d\n", itr);
  }

  printf("\nTotal CPU duration (20 runs): %.0f microseconds\n", cpu_duration);
  printf("Total GPU duration (20 runs): %.0f microseconds\n", gpu_duration);
  printf("Speedup (CPU/GPU): %.2fx\n", cpu_duration / gpu_duration);

  free(h_a);
  free(h_b);
  free(h_c);
  free(h_c_gpu);
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  return 0;
}
