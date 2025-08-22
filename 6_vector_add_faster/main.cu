/*

Benchcmarked the codes for CPU and GPU.

Play with N and see what your graphic card shows you.

*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <math.h>

using namespace std;

#define N 10000000

void init_vector(float *vect, int n) {
  for(int i = 0;i<n;i++) {
    vect[i] = static_cast<float>(rand())/RAND_MAX;
  }
}

double get_time() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + ts.tv_nsec * 1e-9;
}

void vector_add_cpu(float *a, float *b, float *c, int n) {
  for(int i = 0;i<n;i++) {
    c[i] = a[i] + b[i];
  }
}

__global__ void vector_add_gpu(float *a, float *b, float *c, int n) {

  int threadID = blockIdx.x * blockDim.x + threadIdx.x;
  if(threadID < n) {
    c[threadID] = a[threadID] + b[threadID];
  }

}

int main() {

  size_t size = N*sizeof(float);

  int threads_per_block = 256;
  int num_blocks = 1 + (N-1)/threads_per_block;

  float *h_a = (float*)malloc(size);
  float *h_b = (float*)malloc(size);
  float *h_c = (float*)malloc(size);
  float *h_c_gpu = (float*)malloc(size);

  float *d_a, *d_b, *d_c;

  cudaMalloc(&d_a, size);
  cudaMalloc(&d_b, size);
  cudaMalloc(&d_c, size);

  srand(time(NULL));
  init_vector(h_a, N);
  init_vector(h_b, N);

  cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

  printf("Performing warmup runs ...\n");
  for(int i = 0;i<3;i++) {
    vector_add_cpu(h_a, h_b, h_c, N);
    vector_add_gpu<<<num_blocks, threads_per_block>>>(d_a, d_b, d_c, N);
    cudaDeviceSynchronize();
  }


  printf("Benchmarking CPU implementation...\n");

  double cpu_total_time = 0.0;
  for(int i = 0;i<20;i++) {
    double start_time = get_time();
    vector_add_cpu(h_a,h_b,h_c, N);
    double end_time = get_time();
    cpu_total_time += end_time - start_time;
  }
  double cpu_avg_time = cpu_total_time / 20.0;


  printf("Benchmarking GPU implementation...\n");
  double gpu_total_time = 0.0;
  for(int i = 0;i<20;i++) {
    double start_time = get_time();
    vector_add_gpu<<<num_blocks, threads_per_block>>>(d_a,d_b,d_c,N);
    cudaDeviceSynchronize();
    double end_time = get_time();
    gpu_total_time += end_time - start_time;
  }
  double gpu_avg_time = gpu_total_time / 20.0;

  // Print results
  printf("CPU average time: %f milliseconds\n", cpu_avg_time*1000);
  printf("GPU average time: %f milliseconds\n", gpu_avg_time*1000);
  printf("Speedup: %fx\n", cpu_avg_time / gpu_avg_time);


  cudaMemcpy(h_c_gpu, d_c, size, cudaMemcpyDeviceToHost);


  bool cpu_gpu_equal = true;
  for(int i = 0;i<N;i++) {
    if(fabs(h_c_gpu[i] - h_c[i]) > 1e-5) {
      cpu_gpu_equal = false;
      break;
    }
  }

  if(cpu_gpu_equal) printf("CPU and GPU calculations are equal.\n");
  else printf("CPU and GPU calculations are not equal\n");

  free(h_a);
  free(h_b);
  free(h_c);

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  return 0;
}