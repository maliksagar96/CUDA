#include <iostream>
#include <cstdlib>
#include <chrono>
#include <cuda_runtime.h>

using namespace std;

#define N 50000000
#define THREADS_PER_BLOCK 1024

// Kernel: block-level reduction using shared memory
__global__ void gpu_array_sum(double *d_in, double *d_partial, int n) {
    __shared__ double sdata[THREADS_PER_BLOCK];
    
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Each thread loads one element (or 0 if out of bounds)
    double x = (idx < n) ? d_in[idx] : 0.0;
    sdata[tid] = x;
    __syncthreads();

    // Reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    // Thread 0 writes block's sum to global memory
    if (tid == 0)
        d_partial[blockIdx.x] = sdata[0];
}

int main() {
    srand(42);

    // CPU array
    double *array = new double[N];
    for (int i = 0; i < N; i++)
        array[i] = rand() % 100 + 1;

    // CPU baseline sum
    double cpu_sum = 0;
    auto start_cpu = chrono::high_resolution_clock::now();
    for (int i = 0; i < N; i++)
        cpu_sum += array[i];    
    auto end_cpu = chrono::high_resolution_clock::now();
    chrono::duration<double, milli> cpu_time = end_cpu - start_cpu;
    cout << "CPU sum = " << cpu_sum << " , time = " << cpu_time.count() << " ms" << endl;

    // GPU memory
    double *d_array, *d_partial;
    int threads = THREADS_PER_BLOCK;
    int blocks = (N + threads - 1) / threads;

    cudaMalloc((void**)&d_array, N * sizeof(double));
    cudaMalloc((void**)&d_partial, blocks * sizeof(double));

    auto start_total = chrono::high_resolution_clock::now();
    cudaMemcpy(d_array, array, N * sizeof(double), cudaMemcpyHostToDevice);

    // Launch kernel
    gpu_array_sum<<<blocks, threads>>>(d_array, d_partial, N);
    cudaDeviceSynchronize();

    // Copy partial sums back to CPU
    double *partial = new double[blocks];
    cudaMemcpy(partial, d_partial, blocks * sizeof(double), cudaMemcpyDeviceToHost);

    // Final sum on CPU
    double gpu_sum = 0;
    for (int i = 0; i < blocks; i++)
        gpu_sum += partial[i];

    auto end_total = chrono::high_resolution_clock::now();
    chrono::duration<double, milli> gpu_total_time = end_total - start_total;

    cout << "GPU sum = " << gpu_sum 
         << ", total time (mem + kernel + final sum) = " << gpu_total_time.count() << " ms" << endl;

    // Cleanup
    delete[] array;
    delete[] partial;
    cudaFree(d_array);
    cudaFree(d_partial);

    return 0;
}
