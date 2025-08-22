/*

Now this is a simple sample code to use cuda streams. 

Now think streams as multiple pipes from which water is flowing. If you have multiple pipes then you can send water through each pipe simultaneouly and 
increase the parallelism of your code. 

Here the data is divided into chunks so that it can se sent through each stream independently. 

Also the memory transfer from host to device and from device to host is aysnchronous, that means the cpu doesn't wait for the memory transfer. 
It can happen in the background and cpu can start doing other tasks and you can launch other kernels in the meantime.

*/


#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <math.h>

#define N 10000000

void init_vector(float *vect, int n) {
  for(int i = 0; i < n; i++) {
    vect[i] = static_cast<float>(rand()) / RAND_MAX;
  }
}

double get_time() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + ts.tv_nsec * 1e-9;
}

void vector_add_cpu(float *a, float *b, float *c, int n) {
  for(int i = 0; i < n; i++) {
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
  size_t size = N * sizeof(float);

  int numstreams = 10;
  int chunk_size = N/numstreams;
  

  int threads_per_block = 256;
  int num_blocks = (N + threads_per_block - 1) / threads_per_block;

  float *h_a, *h_b, *h_c, *h_c_gpu;
  float *d_a, *d_b, *d_c;

  // Use pinned memory on host
  cudaMallocHost(&h_a, size);
  cudaMallocHost(&h_b, size);
  cudaMallocHost(&h_c, size);
  cudaMallocHost(&h_c_gpu, size);

  // Device memory
  cudaMalloc(&d_a, size);
  cudaMalloc(&d_b, size);
  cudaMalloc(&d_c, size);

  srand(time(NULL));
  init_vector(h_a, N);
  init_vector(h_b, N);

  cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

  printf("Performing warmup runs ...\n");
  for(int i = 0; i < 3; i++) {
    vector_add_cpu(h_a, h_b, h_c, N);
    vector_add_gpu<<<num_blocks, threads_per_block>>>(d_a, d_b, d_c, N);
    cudaDeviceSynchronize();
  }

  printf("Benchmarking CPU implementation...\n");
  double cpu_total_time = 0.0;
  for(int i = 0; i < 20; i++) {
    double start_time = get_time();
    vector_add_cpu(h_a, h_b, h_c, N);
    double end_time = get_time();
    cpu_total_time += end_time - start_time;
  }
  double cpu_avg_time = cpu_total_time / 20.0;

  cudaStream_t stream[numstreams];
  for(int i = 0;i<numstreams;i++) {
    cudaStreamCreate(&stream[i]);
  }

printf("Benchmarking GPU implementation...\n");

float gpu_total_time = 0.0f;

cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

int stream_blocks = 1 + (chunk_size-1)/threads_per_block;

for (int i = 0; i < 20; i++) {
  cudaEventRecord(start, 0);
  for(int s = 0;s<numstreams;s++) {
    int offset = s * chunk_size;
    
    cudaMemcpyAsync(d_a + offset, h_a + offset, chunk_size * sizeof(float), cudaMemcpyHostToDevice, stream[s]);
    cudaMemcpyAsync(d_b + offset, h_b + offset, chunk_size * sizeof(float), cudaMemcpyHostToDevice, stream[s]);
    vector_add_gpu<<<stream_blocks, threads_per_block, 0, stream[s]>>>(d_a + offset, d_b + offset, d_c + offset, chunk_size);
    cudaMemcpyAsync(h_c_gpu + offset, d_c + offset, chunk_size * sizeof(float), cudaMemcpyDeviceToHost, stream[s]);
  }

  for(int s = 0;s<numstreams;s++) {
    cudaStreamSynchronize(stream[s]);
  }
  
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);

  float milliseconds = 0.0f;
  cudaEventElapsedTime(&milliseconds, start, stop);
  gpu_total_time += milliseconds;
}

cudaEventDestroy(start);
cudaEventDestroy(stop);

float gpu_avg_time = gpu_total_time / 20.0f;

printf("CPU average time: %f milliseconds\n", cpu_avg_time * 1000);
printf("GPU average time: %f milliseconds\n", gpu_avg_time);
printf("Speedup: %fx\n", (cpu_avg_time * 1000) / gpu_avg_time);

  bool cpu_gpu_equal = true;
  for(int i = 0; i < N; i++) {
    if(fabs(h_c_gpu[i] - h_c[i]) > 1e-5) {
      cpu_gpu_equal = false;
      break;
    }
  }

  if(cpu_gpu_equal)
    printf("CPU and GPU calculations are equal.\n");
  else
    printf("CPU and GPU calculations are not equal\n");

  for(int i = 0;i<numstreams;i++) {
    cudaStreamDestroy(stream[i]); 
  }

  // Free pinned host memory
  cudaFreeHost(h_a);
  cudaFreeHost(h_b);
  cudaFreeHost(h_c);
  cudaFreeHost(h_c_gpu);

  // Free device memory
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  return 0;
}
