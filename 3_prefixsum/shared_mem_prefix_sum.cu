/*
  This is actually a great code example to write and understand the concept of shared memory.
  The logic is similar to binary addition.

  1) First, we copy the array into shared memory, so all threads in a block can access it quickly.

  2) Then we perform additions in multiple passes.

  3) In the first pass, each thread adds the element just before it.

  4) In the next pass, it adds the element two places before, then four, and so on.

Try working this out with pen and paper — you’ll clearly see how the prefix sum builds up step by step.

To better understand what's happening in the first pass, I’ve written a hardcoded stride = 1 version of the kernel. It helps isolate and observe exactly how the first stage of addition works.
*/

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void shared_mem_prefix_sum(int* d_a, int* d_prefix_sum, int n) {
    __shared__ int temp[1024];
    int tid = threadIdx.x;

    if (tid < n)
        temp[tid] = d_a[tid];
    __syncthreads();

    for (int stride = 1; stride < n; stride *= 2) {
        int val = 0;
        if (tid >= stride)
            val = temp[tid - stride];

        __syncthreads();
        temp[tid] += val;
        __syncthreads();
    }

    if (tid < n)
        d_prefix_sum[tid] = temp[tid];
}

// __global__ void shared_mem_prefix_sum_stride1(int* d_a, int* d_prefix_sum, int n) {
//     __shared__ int temp[1024];
//     int tid = threadIdx.x;

//     if (tid < n)
//         temp[tid] = d_a[tid];
//     __syncthreads();

//     // Only perform stride = 1
//     int val = 0;
//     if (tid >= 1)
//         val = temp[tid - 1];

//     __syncthreads();
//     temp[tid] += val;
//     __syncthreads();

//     if (tid < n)
//         d_prefix_sum[tid] = temp[tid];
// }

int main() {
    int n = 10;
    int* a = (int*)malloc(sizeof(int) * n);
    int* prefix_sum = (int*)malloc(sizeof(int) * n);

    int *d_a, *d_prefix_sum;

    cudaMalloc((void**)&d_a, sizeof(int) * n);
    cudaMalloc((void**)&d_prefix_sum, sizeof(int) * n);

    for (int i = 0; i < n; i++) {
        a[i] = i + 1;  // a = {1, 2, ..., 10}
    }

    cudaMemcpy(d_a, a, sizeof(int) * n, cudaMemcpyHostToDevice);

    shared_mem_prefix_sum_stride1<<<1, n>>>(d_a, d_prefix_sum, n);
    cudaDeviceSynchronize();  // ensure kernel is done

    cudaMemcpy(prefix_sum, d_prefix_sum, sizeof(int) * n, cudaMemcpyDeviceToHost);

    for (int i = 0; i < n; i++) {
        printf("prefix_sum[%d] = %d\n", i, prefix_sum[i]);
    }

    free(a);
    free(prefix_sum);
    cudaFree(d_a);
    cudaFree(d_prefix_sum);

    return 0;
}
